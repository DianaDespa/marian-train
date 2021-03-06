#pragma once

#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "3rd_party/threadpool.h"
#include "common/definitions.h"
#include "data/batch_generator.h"
#include "optimizers/optimizers.h"
#include "training/dropper.h"
#include "training/scheduler.h"
#include "training/sparse_tensor.h"
#include "training/training.h"
#include "training/validator.h"
#include "quantization/quantized_tensor.h"
#include "quantization/quantized_tensor_32_bit_indices.h"
#include "quantization/quantized_tensor_32_bit_compressed.h"

namespace marian {

class GraphGroup {
protected:
  Ptr<Config> options_;
  Ptr<OptimizerBase> opt_;
  bool scaleLR; // Whether to scale the learning rate
  float averageBatchWords;

public:
  GraphGroup(Ptr<Config> options)
      : options_(options),
      opt_(Optimizer(options)),
      scaleLR(options->get<bool>("batch-flexible-lr")),
      averageBatchWords(options->get<float>("batch-normal-words")) {}

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch>) = 0;

  virtual void load() = 0;

  virtual void save(bool = false) = 0;

  virtual Ptr<data::BatchStats> collectStats() = 0;
};

template <class Builder>
class SingletonGraph : public GraphGroup {
public:
  typedef Builder builder_type;
  typedef typename Builder::dataset_type dataset_type;

  virtual void setScheduler(Ptr<Scheduler<dataset_type>> scheduler) {
    scheduler_ = scheduler;
    // optimizer has to be registered last to see a change of learning rate
    scheduler_->registerTrainingObserver(scheduler_);
    scheduler_->registerTrainingObserver(opt_);
  }

private:
  Ptr<Builder> builder_;
  Ptr<ExpressionGraph> graph_;

  Ptr<Scheduler<dataset_type>> scheduler_;

  Ptr<ExpressionGraph> mvAvgGraph_;
  bool mvAvg_{false};
  float mvDecay_{0.9999};

  void updateMovingAverage(Tensor mvAvgParams, Tensor params, size_t batches) {
    float decay = min(mvDecay_, (float)(batches + 1) / (float)(batches + 10));
    Element(_1 = (decay * _1) + ((1.f - decay) * _2), mvAvgParams, params);
  }

  void execute(Ptr<data::Batch> batch) {
    auto costNode = builder_->build(graph_, batch);

    graph_->forward();
    float cost = costNode->scalar();
    graph_->backward();

    //Get batch stats
    size_t batchWords = batch->words();
    //@TODO use this to gather statistics about the usual number of words per batch
    //std::cout << "Batch size: " << batch->size() << " batchWords " << batchWords << std::endl;

    if (scaleLR) {
      opt_->update(graph_, batchWords/averageBatchWords);
    } else {
      opt_->update(graph_);
    }

    if(mvAvg_) {
      if(!mvAvgGraph_) {
        mvAvgGraph_ = New<ExpressionGraph>();
        mvAvgGraph_->setDevice(graph_->getDevice());
        mvAvgGraph_->copyParams(graph_);
      } else {
        updateMovingAverage(mvAvgGraph_->params()->vals(),
                            graph_->params()->vals(),
                            scheduler_->numberOfBatches());
      }
    }

    if(scheduler_) {
      scheduler_->update(cost, batch);

      if(scheduler_->saving())
        this->save();

      if(scheduler_->validating()) {
        if(mvAvg_)
          scheduler_->validate(mvAvgGraph_);
        else
          scheduler_->validate(graph_);
      }

      /*if(mvAvg_) {
        size_t injectFreq = options_->get<size_t>("moving-inject-freq");
        if(injectFreq && scheduler_->numberOfBatches() % injectFreq == 0) {
          // LOG(info)->info("{} : Injecting moving average into training parameters",
                          scheduler_->numberOfBatches());
          graph_->params()->vals()->copyFrom(mvAvgGraph_->params()->vals());
        }
      }*/
    }
  }

public:
  template <class... Args>
  SingletonGraph(Ptr<Config> options, Args... args)
      : GraphGroup(options),
        mvAvg_{options_->get<bool>("moving-average")},
        mvDecay_{(float)options_->get<double>("moving-decay")} {
    size_t device = options_->get<std::vector<size_t>>("devices")[0];

    graph_ = New<ExpressionGraph>();
    graph_->setDevice(device);
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    opt_ = Optimizer(options_);

    builder_ = New<Builder>(options_, args...);
  }

  void update(Ptr<data::Batch> batch) { execute(batch); }

  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(boost::filesystem::exists(name)) {
        if(scheduler_)
          scheduler_->load(name);
        builder_->load(graph_, name);
      }
    }
  }

  void save(bool final = false) {
    auto saveGraph = graph_;
    if(mvAvg_)
      saveGraph = mvAvgGraph_;

    save(saveGraph, final);
  }

  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      builder_->save(graph_, name, true);
      if(scheduler_)
        scheduler_->save(name);
    } else {
      std::string name = options_->get<std::string>("model");

      if(!final) {
        std::string numberOfBatches
            = scheduler_ ? std::to_string(scheduler_->numberOfBatches()) :
                           "unknown";
        std::string nameOverwrite = name;
        nameOverwrite.replace(
            name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
        builder_->save(graph_, nameOverwrite);
      }

      builder_->save(graph_, name, true);
      if(scheduler_)
        scheduler_->save(name);
    }
  }

  Ptr<data::BatchStats> collectStats() {
    return builder_->collectStats(graph_);
  }
};

template <class Builder>
class AsyncGraphGroup : public GraphGroup {
public:
  typedef Builder builder_type;
  typedef typename Builder::dataset_type dataset_type;

  virtual void setScheduler(Ptr<Scheduler<dataset_type>> scheduler) {
    scheduler_ = scheduler;
    // optimizer has to be registered last to see a change of learning rate
    scheduler_->registerTrainingObserver(scheduler_);
    scheduler_->registerTrainingObserver(opt_);
  }

private:
  bool first_{true};

  std::vector<Ptr<Builder>> builders_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<size_t> devices_;

  Ptr<Scheduler<dataset_type>> scheduler_;

  std::mutex sync_;
  std::vector<std::mutex> shardSync_;

  boost::shared_mutex schedulerMutex_;

  std::vector<SparseTensor> localSparseGrads_;
  std::vector<SparseTensor> sparseGrads_;
  std::vector<SparseTensor> tmpSparseDelta;
  std::vector<std::vector<SparseTensor>> localSparseDelta;

  std::vector<QuantizedTensor> localQuantizedGrads_;
  std::vector<QuantizedTensor> quantizedGrads_;
  std::vector<QuantizedTensor> tmpQuantizedDelta;
  std::vector<std::vector<QuantizedTensor>> localQuantizedDelta;

  // version number per-shard
  std::vector<int> globalVersionNumber;

  // each worker has the version number obtained from each shard
  std::vector<std::vector<int>> localVersionNumbers;

  std::vector<std::vector<GradientDrop>> fetchDropper;
  std::vector<Tensor> tmpTensor;

  std::vector<std::vector<Tensor>> params_;

  std::vector<Tensor> grads_;

  std::vector<Ptr<OptimizerBase>> shardOpt_;

  int shardSize_;

  std::vector<Tensor> paramsAvg_;

  bool movingAvg_{false};
  float mvDecay_{0.9999};

  ThreadPool pool_;

  double dropRate_{0};
  int historySize_{1};

  size_t tau_{1};

  int quantizationVariant_;
  static const int QUANTIZATION_SIMULATED{0};
  static const int QUANTIZATION_32_BIT_INDICES{1};
  static const int QUANTIZATION_32_BIT_COMPRESSED{2};

  std::vector<Ptr<TensorAllocator>> allocators_;

  Tensor newTensor(int size, int device) {
    Tensor t;
    Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);
    allocator->reserveExact(size * sizeof(float));
    allocator->allocate(t, {1, size});
    allocators_.push_back(allocator);

    return t;
  }

  void initQuantizedVars(int sparseCapacity) {
    if(quantizationVariant_ == QUANTIZATION_32_BIT_INDICES) {

      for(auto device : devices_) {
        quantizedGrads_.push_back(
            QuantizedTensor(new QuantizedTensor32BitIndices(device)));
        localQuantizedGrads_.push_back(
            QuantizedTensor(new QuantizedTensor32BitIndices(device)));
        tmpQuantizedDelta.push_back(QuantizedTensor(
            new QuantizedTensor32BitIndices(device)));
        std::vector<QuantizedTensor> tmp;
        for(int i = 0; i < devices_.size(); i++)
          tmp.push_back(QuantizedTensor(
              new QuantizedTensor32BitIndices(device)));
        localQuantizedDelta.push_back(tmp);
      }
    } else if(quantizationVariant_ == QUANTIZATION_32_BIT_COMPRESSED) {

      for(auto device : devices_) {
        quantizedGrads_.push_back(
            QuantizedTensor(new QuantizedTensor32BitCompressed(sparseCapacity, device)));
        localQuantizedGrads_.push_back(
            QuantizedTensor(new QuantizedTensor32BitCompressed(sparseCapacity, device)));
        tmpQuantizedDelta.push_back(QuantizedTensor(
            new QuantizedTensor32BitCompressed(sparseCapacity / devices_.size(), device)));
        std::vector<QuantizedTensor> tmp;
        for(int i = 0; i < devices_.size(); i++)
          tmp.push_back(QuantizedTensor(
              new QuantizedTensor32BitCompressed(sparseCapacity / devices_.size(), device)));
        localQuantizedDelta.push_back(tmp);
      }
    }
  }

  void fetchParams(Tensor oldParams, const std::vector<Tensor>& params) {
    // @TODO read guard on parameters
    int pos = 0;

    std::vector<std::thread> threads;
    for(int idx = 0; idx < devices_.size(); idx++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            oldParams->subtensor(pos, params[idx]->size())
                ->copyFrom(params[idx]);
          },
          idx,
          pos));

      pos += shardSize_;
    }
    for(auto&& t : threads) {
      t.join();
    }
  }

  void pushGradients(Tensor newGrads, size_t batchWords) {
    // add instead of copy?
    std::vector<std::thread> threads;
    int pos = 0;
    for(int idx = 0; idx < devices_.size(); idx++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            grads_[idx]->copyFrom(
                newGrads->subtensor(pos, grads_[idx]->size()));

            // apply and increment your version number, if history is enabled
            int latestVersion = 0;

            if(historySize_ > 1) {
              int pastVersion = globalVersionNumber[idx] % historySize_;
              latestVersion = ++globalVersionNumber[idx] % historySize_;
              params_[latestVersion][idx]->copyFrom(params_[pastVersion][idx]);
            }

            if (scaleLR) {
              shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx], batchWords/averageBatchWords);
            } else {
              shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx]);
            }

            if(movingAvg_)
              updateMovingAverage(paramsAvg_[idx], params_[latestVersion][idx],
                                  scheduler_->numberOfBatches());
          },
          idx,
          pos));

      pos += shardSize_;
    }
    for(auto&& t : threads)
      t.join();
  }

  void sparseFetchParams(Tensor oldParams, int workerId) {
    if(graphs_.size() < 2)
      return;

    // @TODO read guard on parameters
    int p = 0;

    std::vector<std::thread> threads;
    for(int i = 0; i < devices_.size(); i++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            // obtain the delta
            int latestVersion = globalVersionNumber[idx] % historySize_;
            int currVersion
                = localVersionNumbers[workerId][idx] % historySize_;

            // check if the current version is too old
            if(globalVersionNumber[idx] - localVersionNumbers[workerId][idx]
               >= historySize_)
              currVersion = (1 + globalVersionNumber[idx])
                            % historySize_;  // if so, pick the best you can do

            // if already latest
            if(globalVersionNumber[idx] == localVersionNumbers[workerId][idx])
              return;

            // get delta : param latest version - current param (locally)
            Element(_1 = _2 - _3,
                    tmpTensor[idx],
                    params_[latestVersion][idx],
                    params_[currVersion][idx]);

            // get sparse delta
            fetchDropper[workerId][idx]->dropGraph(
                tmpTensor[idx], tmpSparseDelta[idx], dropRate_);

            // move sparse delta
            localSparseDelta[workerId][idx]->copyFrom(tmpSparseDelta[idx]);

            localSparseDelta[workerId][idx]->scatterAdd(
                oldParams->subtensor(pos, grads_[idx]->size()));

            localVersionNumbers[workerId][idx] = globalVersionNumber[idx];
          },
          i,
          p));

      p += shardSize_;
    }
    for(auto&& t : threads) {
      t.join();
    }
  }

  void sparsePushGradients(SparseTensor newGrads, size_t batchWords) {
    if(graphs_.size() < 2) {
      if (scaleLR) {
        opt_->update(graphs_[0], batchWords/averageBatchWords);
      } else {
        opt_->update(graphs_[0]);
      }
    } else {
      // add instead of copy?
      std::vector<std::thread> threads;
      int pos = 0;
      for(int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back(std::thread(
            [=](int idx, int pos) {
              // individual mutex per-shard
              std::lock_guard<std::mutex> guard(shardSync_[idx]);

              // send shard
              sparseGrads_[idx]->copyFrom(newGrads->subtensor(pos, grads_[idx]->size(), idx));

              // convert back to dense, with index offset of -pos
              sparseGrads_[idx]->toDense(grads_[idx], -pos);

              // apply and increment your version number
              int pastVersion = globalVersionNumber[idx] % historySize_;
              int latestVersion = ++globalVersionNumber[idx] % historySize_;
              params_[latestVersion][idx]->copyFrom(params_[pastVersion][idx]);
              if (scaleLR) {
                shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx], batchWords/averageBatchWords);
              } else {
                shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx]);
              }

              if(movingAvg_)
                updateMovingAverage(paramsAvg_[idx],
                                    params_[latestVersion][idx],
                                    scheduler_->numberOfBatches());

            },
            idx,
            pos));

        pos += shardSize_;
      }
      for(auto&& t : threads)
        t.join();
    }
  }

  void quantizedFetchParams(Tensor oldParams, int workerId) {
    // LOG(info)->info("Fetch");
    if(graphs_.size() < 2)
      return;

    // @TODO read guard on parameters
    int p = 0;

    std::vector<std::thread> threads;
    for(int i = 0; i < devices_.size(); i++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            // obtain the delta
            int latestVersion = globalVersionNumber[idx] % historySize_;
            int currVersion
                = localVersionNumbers[workerId][idx] % historySize_;

            // check if the current version is too old
            if(globalVersionNumber[idx] - localVersionNumbers[workerId][idx]
               >= historySize_)
              currVersion = (1 + globalVersionNumber[idx])
                            % historySize_;  // if so, pick the best you can do

            // if already latest
            if(globalVersionNumber[idx] == localVersionNumbers[workerId][idx])
              return;

            // get delta : param latest version - current param (locally)
            Element(_1 = _2 - _3,
                    tmpTensor[idx],
                    params_[latestVersion][idx],
                    params_[currVersion][idx]);

            // get sparse delta
            tmpQuantizedDelta[idx]->encode(tmpTensor[idx]);

            // move sparse delta
            localQuantizedDelta[workerId][idx]->copyFrom(tmpQuantizedDelta[idx]);

            localQuantizedDelta[workerId][idx]->scatterAdd(
                oldParams->subtensor(pos, grads_[idx]->size()));

          },
          i,
          p));

      p += shardSize_;
    }
    for(auto&& t : threads) {
      t.join();
    }
  }

  void quantizedPushGradients(QuantizedTensor newGrads, size_t batchWords) {
    // LOG(info)->info("Push");    
    if(graphs_.size() < 2) {
      if (scaleLR) {
        opt_->update(graphs_[0], batchWords/averageBatchWords);
      } else {
        opt_->update(graphs_[0]);
      }
    } else {
      // add instead of copy?
      std::vector<std::thread> threads;
      int pos = 0;
      for(int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back(std::thread(
            [=](int idx, int pos) {
              // individual mutex per-shard
              std::lock_guard<std::mutex> guard(shardSync_[idx]);

              // send shard
              quantizedGrads_[idx]->copyFrom(newGrads->subtensor(pos, grads_[idx]->size()));

              // convert back to dense, with index offset of -pos
              quantizedGrads_[idx]->decode(grads_[idx], -pos);

              // apply and increment your version number
              int pastVersion = globalVersionNumber[idx] % historySize_;
              int latestVersion = ++globalVersionNumber[idx] % historySize_;
              params_[latestVersion][idx]->copyFrom(params_[pastVersion][idx]);
              
              if (scaleLR) {
                shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx], batchWords/averageBatchWords);
              } else {
                shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx]);
              }

              if(movingAvg_)
                updateMovingAverage(paramsAvg_[idx],
                                    params_[latestVersion][idx],
                                    scheduler_->numberOfBatches());

            },
            idx,
            pos));

        pos += shardSize_;
      }
      for(auto&& t : threads)
        t.join();
    }
  }

  void updateMovingAverage(Tensor paramsAvg, Tensor params, size_t batches) {
    float decay = min(mvDecay_, (float)(batches + 1) / (float)(batches + 10));
    Element(_1 = (decay * _1) + ((1.f - decay) * _2), paramsAvg, params);
  }

  void execute(Ptr<data::Batch> batch) {
    if(first_) {
      // initialize the parameters
      for(size_t i = 0; i < graphs_.size(); ++i) {
        // takes care of thead_local stuff
        THREAD_GUARD(builders_[i]->build(graphs_[i], batch);
                     graphs_[i]->forward(););

        globalVersionNumber.push_back(0);
        std::vector<int> localVersion;
        for(int j = 0; j < graphs_.size(); j++)
          localVersion.push_back(0);

        localVersionNumbers.push_back(localVersion);
      }

      if(params_[0].size() == 0) {
        int totalSize = graphs_[0]->params()->vals()->size();
        shardSize_ = ceil(totalSize / devices_.size());

        int pos = 0;
        // parameter sharding
        for(auto device : devices_) {
          int __size__ = min(shardSize_, totalSize);
          totalSize -= __size__;

          for(int hId = 0; hId < historySize_; hId++) {
            Tensor param = newTensor(__size__, device);
            param->copyFrom(
                graphs_[0]->params()->vals()->subtensor(pos, __size__));
            params_[hId].push_back(param);
          }

          if(dropRate_)
            tmpTensor.push_back(newTensor(__size__, device));
          pos += __size__;
        }
      }
      if(grads_.size() == 0) {
        int totalSize = graphs_[0]->params()->vals()->size();

        for(auto device : devices_) {
          int __size__ = min(shardSize_, totalSize);
          totalSize -= __size__;
          Tensor grad_ = newTensor(__size__, device);
          grads_.push_back(grad_);
        }
      }
      if(movingAvg_) {
        if(paramsAvg_.size() == 0) {
          int totalSize = graphs_[0]->params()->vals()->size();

          int i = 0;
          for(auto device : devices_) {
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;
            Tensor paramAvg = newTensor(__size__, device);

            paramAvg->copyFrom(params_[0][i++]);
            paramsAvg_.push_back(paramAvg);
          }
        }
      }

      if(dropRate_ && first_) {
        int totalSize = graphs_[0]->params()->vals()->size();
        int sparseCapacity = totalSize * 1.2 * (1.0 - dropRate_);
        if (quantizationVariant_ == QUANTIZATION_SIMULATED) {
          for(auto device : devices_) {
            sparseGrads_.push_back(
                SparseTensor(new SparseTensorBase(sparseCapacity, device)));
            localSparseGrads_.push_back(
                SparseTensor(new SparseTensorBase(sparseCapacity, device)));
            tmpSparseDelta.push_back(SparseTensor(
                new SparseTensorBase(sparseCapacity / devices_.size(), device)));
            std::vector<SparseTensor> tmp;
            for(int i = 0; i < devices_.size(); i++)
              tmp.push_back(SparseTensor(
                  new SparseTensorBase(sparseCapacity / devices_.size(), device)));
            localSparseDelta.push_back(tmp);
          }
        } else {
          initQuantizedVars(sparseCapacity);
        }
      }

      first_ = false;
    }

    auto task = [this](Ptr<data::Batch> batch) {
      static size_t i = 0;
      thread_local Ptr<ExpressionGraph> graph;
      thread_local Ptr<Builder> builder;
      thread_local size_t t = 0;
      thread_local size_t numSeenWords = 0;

      thread_local Tensor accGradients;
      thread_local Ptr<TensorAllocator> accAlloc;

      // gradient drop purpose
      thread_local GradientDrop dropper;

      thread_local size_t myId = 0;

      std::vector<std::pair<int,int>> layerShapes;

      if(!graph) {
        std::lock_guard<std::mutex> lock(sync_);
        myId = i;
        graph = graphs_[i];
        builder = builders_[i++];

        for (auto& x: graph->params()->getMap()) {
          layerShapes.push_back({x.second->shape()[0], x.second->shape()[1]});
        }
      }

      if(!dropper) {
        std::lock_guard<std::mutex> lock(sync_);
        dropper = GradientDrop(new GradientDropBase(
            options_->get<int>("quantize-bits"),
            options_->get<bool>("quantize-min-drop"),
            options_->get<bool>("quantize-column-wise")
        ));
        std::vector<GradientDrop> tmp;
        for(int i = 0; i < devices_.size(); i++)
          tmp.push_back(GradientDrop(new GradientDropBase(
              options_->get<int>("quantize-bits"),
              options_->get<bool>("quantize-min-drop"),
              options_->get<bool>("quantize-column-wise")
          )));
        fetchDropper.push_back(tmp);
      }

      auto costNode = builder->build(graph, batch);

      if(t % tau_ == 0) {

        if(dropRate_ && t > 0)
          if(quantizationVariant_ == QUANTIZATION_SIMULATED)
            sparseFetchParams(graph->params()->vals(), myId);
          else
            quantizedFetchParams(graph->params()->vals(), myId);
        else
          fetchParams(graph->params()->vals(),
                      params_[globalVersionNumber[myId] % historySize_]);

      }

      graph->forward();
      float cost = costNode->scalar();
      graph->backward();

      //Get batch stats
      size_t batchWords = batch->words();

      Tensor gradients;
      if(tau_ > 1) {
        if(t == 0) {
          accAlloc = New<TensorAllocator>(graph->getDevice());
          accAlloc->reserveExact(graph->params()->grads()->memory()->size());
          accAlloc->allocate(accGradients, graph->params()->grads()->shape());
          accGradients->set(0);
        }

        Element(_1 += _2, accGradients, graph->params()->grads());
        gradients = accGradients;
        numSeenWords += batchWords; //Keep track of how many words we've calculated the error from
      }
      else {
        gradients = graph->params()->grads();
        numSeenWords = batchWords;
      }

      t++;

      if(t % tau_ == 0) {
        if(dropRate_) {
          if(quantizationVariant_ == QUANTIZATION_SIMULATED) {
            dropper->dropGraph(
                graph->params()->grads(), localSparseGrads_[myId], dropRate_, layerShapes);
            sparsePushGradients(localSparseGrads_[myId], numSeenWords);
          } else {
            localQuantizedGrads_[myId]->encode(graph->params()->grads());
            quantizedPushGradients(localQuantizedGrads_[myId], numSeenWords);
          }
        } else {
          pushGradients(graph->params()->grads(), numSeenWords);
        }
        numSeenWords = 0; //Reset the counter of seen words after gradient update

        if(tau_ > 1) {
          gradients->set(0);
        }

      }

      if(scheduler_) {
        boost::upgrade_lock<boost::shared_mutex> lock(schedulerMutex_);
        {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          scheduler_->update(cost, batch);
        }

        if(scheduler_->saving()) {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          if(movingAvg_)
            fetchParams(graph->params()->vals(), paramsAvg_);
          this->save(graph);
        }

        if(scheduler_->validating()) {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          if(movingAvg_)
            fetchParams(graph->params()->vals(), paramsAvg_);
          scheduler_->validate(graph);
        }

        /*if(movingAvg_) {
          size_t injectFreq = options_->get<size_t>("moving-inject-freq");
          if(injectFreq && scheduler_->numberOfBatches() % injectFreq == 0) {
            boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);

            // LOG(info)->info("{} : Injecting moving average into training parameters",
                            scheduler_->numberOfBatches());
            for(int idx = 0; idx < paramsAvg_.size(); idx++) {
              std::lock_guard<std::mutex> guard(shardSync_[idx]);
              params_[myId][idx]->copyFrom(paramsAvg_[idx]);
            }
          }
        }*/
      }
    };

    pool_.enqueue(task, batch);
  }

public:
  template <class... Args>
  AsyncGraphGroup(Ptr<Config> options, Args... args)
      : GraphGroup(options),
        devices_{options_->get<std::vector<size_t>>("devices")},
        pool_{devices_.size(), devices_.size()},
        shardSync_{devices_.size()},
        movingAvg_{options_->get<bool>("moving-average")},
        mvDecay_{(float)options_->get<double>("moving-decay")},
        dropRate_{options_->get<double>("drop-rate")},
        tau_{options_->get<size_t>("tau")},
        quantizationVariant_{options_->get<int>("quantize-variant")} {
    if(dropRate_ > 0.0) {
      historySize_ = 1.5 * devices_.size();
    }
    for(int i = 0; i < historySize_; i++)
      params_.push_back(std::vector<Tensor>());
    for(auto device : devices_) {
      auto graph = New<ExpressionGraph>();
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);
      shardOpt_.push_back(Optimizer(options_));
      builders_.push_back(New<Builder>(options_, args...));
    }
  }

  void update(Ptr<data::Batch> batch) { execute(batch); }

  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string init = options_->get<std::string>("model");
      if(boost::filesystem::exists(init)) {
        size_t i = 0;
        if(scheduler_)
          scheduler_->load(init);
        for(auto graph : graphs_)
          builders_[i++]->load(graph, init);
      }
    }
  }

  void save(bool final = false) { save(graphs_[0], final); }

  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    int idx = 0;
    for(int i = 0; i < graphs_.size(); ++i) {
      if(graph == graphs_[i]) {
        idx = i;
        break;
      }
    }

    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    } else {
      std::string name = options_->get<std::string>("model");

      if(!final) {
        std::string numberOfBatches
            = scheduler_ ? std::to_string(scheduler_->numberOfBatches()) :
                           "unknown";
        std::string nameOverwrite = name;
        nameOverwrite.replace(
            name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
        builders_[idx]->save(graphs_[idx], nameOverwrite);
      }

      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    }
  }

  Ptr<data::BatchStats> collectStats() {
    return builders_[0]->collectStats(graphs_[0]);
  }
};
}
