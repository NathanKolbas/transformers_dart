import 'package:transformers/src/base/processing_utils.dart';
import 'package:transformers/src/models/idefics3/processing_idefics3.dart';

final Map<
    String,
    Processor Function(Map<String, dynamic> config, Map<String, dynamic> components)
> AllProcessors = {
  'Idefics3Processor': (config, components) => Idefics3Processor(config, components),
};

final Map<
    String,
    Future<Processor> Function(String pretrained_model_name_or_path, [ProcessorPretrainedOptions? options])
> AllProcessorsFromPretrained = {
  'Idefics3Processor': Idefics3Processor.from_pretrained,
};