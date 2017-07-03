// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

#include <boost/program_options.hpp>

#include <sys/ioctl.h>
#include <unistd.h>
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/file_stream.h"
#include "common/logging.h"

namespace marian {

// try to determine the width of the terminal
uint16_t guess_terminal_width(uint16_t max_width = 180);

class Config {
public:
  static size_t seed;

  Config(int argc,
         char** argv,
         bool validate = true,
         bool translate = false,
         bool rescore = false)
      : cmdline_options_("Allowed options", guess_terminal_width()) {
    addOptions(argc, argv, validate, translate, rescore);
    log();
  }

  Config(const Config& other)
      : config_(YAML::Clone(other.config_)),
        modelFeatures_(other.modelFeatures_) {}

  bool has(const std::string& key) const;

  YAML::Node get(const std::string& key) const;

  template <typename T>
  T get(const std::string& key) const {
    return config_[key].as<T>();
  }

  template <typename T>
  void set(const std::string& key, const T& value) {
    config_[key] = value;
  }

  const YAML::Node& get() const;
  YAML::Node& get();

  YAML::Node operator[](const std::string& key) const { return get(key); }

  void override(const YAML::Node& params);
  YAML::Node getModelParameters();
  void loadModelParameters(const std::string& name);
  void saveModelParameters(const std::string& name);

  void GetYamlFromNpz(YAML::Node&, const std::string&, const std::string&);

  void AddYamlToNpz(const YAML::Node&, const std::string&, const std::string&);

  void addOptions(
      int argc, char** argv, bool validate, bool translate, bool rescore);

  void addOptionsCommon(boost::program_options::options_description&, bool);
  void addOptionsModel(boost::program_options::options_description&, bool, bool);
  void addOptionsTraining(boost::program_options::options_description&);
  void addOptionsRescore(boost::program_options::options_description&);
  void addOptionsValid(boost::program_options::options_description&);
  void addOptionsTranslate(boost::program_options::options_description&);
  void addOptionsSparceMatrixEncoding(boost::program_options::options_description&);

  void log();
  void validateOptions(bool translate = false, bool rescore = false) const;

  void OutputRec(const YAML::Node node, YAML::Emitter& out) const;

  template <class OStream>
  friend OStream& operator<<(OStream& out, const Config& config) {
    YAML::Emitter outYaml;
    config.OutputRec(config.get(), outYaml);
    out << outYaml.c_str();
    return out;
  }

  void save(const std::string& name) {
    OutputFileStream out(name);
    (std::ostream&)out << *this;
  }

private:
  boost::program_options::options_description cmdline_options_;
  YAML::Node config_;
  std::vector<std::string> modelFeatures_;
};
}
