import 'package:transformers/src/base/image_processing_utils.dart';
import 'package:transformers/src/base/processing_utils.dart';
import 'package:transformers/src/models/auto/image_processing_auto.dart';
import 'package:transformers/src/utils/core.dart';
import 'package:transformers/src/utils/hub.dart';
import 'package:transformers/transformers.dart';

/// Prompt with expanded image tokens for when the image is split into patches.
/// @private
String _prompt_split_image(
    int image_seq_len,
    int image_rows,
    int image_cols,
    String fake_token_around_image,
    String image_token,
    String global_img_token,
    ) {
  String text_split_images = "";
  for (int n_h = 0; n_h < image_rows; ++n_h) {
    for (int n_w = 0; n_w < image_cols; ++n_w) {
      text_split_images +=
      '$fake_token_around_image'
          '<row_${n_h + 1}_col_${n_w + 1}>'
          '${List.filled(image_seq_len, image_token).join()}';
    }
    text_split_images += "\n";
  }

  text_split_images +=
  '\n$fake_token_around_image'
      '$global_img_token'
      '${List.filled(image_seq_len, image_token).join()}'
      '$fake_token_around_image';
  return text_split_images;
}

/// Prompt with expanded image tokens for a single image.
/// @private
String _prompt_single_image(
    int image_seq_len,
    String fake_token_around_image,
    String image_token,
    String global_img_token,
    ) => '$fake_token_around_image'
    '$global_img_token'
    '${List.filled(image_seq_len, image_token).join()}'
    '$fake_token_around_image';

String get_image_prompt_string(
    int image_rows,
    int image_cols,
    int image_seq_len,
    String fake_token_around_image,
    String image_token,
    String global_img_token,
    ) {
  if (image_rows == 0 && image_cols == 0) {
    return _prompt_single_image(
      image_seq_len,
      fake_token_around_image,
      image_token,
      global_img_token,
    );
  }

  return _prompt_split_image(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_img_token,
  );
}

class Idefics3Processor extends Processor {
  static final ProcessorReflection reflection = ProcessorReflection(
    constructor: Idefics3Processor.new,
    image_processor_class: ImageProcessorClassReflection(
      from_pretrained: AutoImageProcessor.from_pretrained,
    ),
    tokenizer_class: TokenizerClassReflection(
      from_pretrained: AutoTokenizer.from_pretrained,
    ),
    uses_processor_config: true,
  );

  final String fake_image_token = '<fake_token_around_image>';
  final String image_token = '<image>';
  final String global_img_token = '<global-img>';

  Idefics3Processor(super.config, super.components);

  ///
  /// @param {string|string[]} text
  /// @param {RawImage|RawImage[]|RawImage[][]} images
  /// @returns {Promise<any>}
  @override
  Future<dynamic> call(dynamic text, [List<dynamic> args = const []]) async {
    final [dynamic images, Map<String, dynamic> options] = args;
    options['return_row_col_info'] ??= true;

    ImageProcessorResult? image_inputs;

    if (images != null) {
      image_inputs = await image_processor!(images, [options]);
    }

    // NOTE: We assume text is present
    if (text is! List) {
      text = [text];
    }

    final List<List<int>> image_rows = image_inputs?.rows ?? [List.filled(text.length, 0)];
    final List<List<int>> image_cols = image_inputs?.cols ?? [List.filled(text.length, 0)];

    final image_seq_len = config['image_seq_len'];
    final n_images_in_text = [];
    final List<String> prompt_strings = [];
    for (int i = 0; i < text.length; ++i) {
      final sample = text[i];
      final sample_rows = image_rows[i];
      final sample_cols = image_cols[i];

      n_images_in_text.add(countString(sample, image_token));

      // Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
      final image_prompt_strings = sample_rows.indexed.map((x) {
        final (j, n_rows) = x;
        return get_image_prompt_string(
          n_rows,
          sample_cols[j],
          image_seq_len,
          fake_image_token,
          image_token,
          global_img_token,
        );
      }).toList(growable: false);

      final split_sample = sample.split(image_token);
      if (split_sample.isEmpty) {
        throw StateError('The image token should be present in the text.');
      }

      // Place in the image prompt strings where the image tokens are
      String new_sample = split_sample[0];
      for (int j = 0; j < image_prompt_strings.length; ++j) {
        new_sample += image_prompt_strings[j] + split_sample[j + 1];
      }
      prompt_strings.add(new_sample);
    }

    final BatchEncoding text_inputs = await tokenizer!(prompt_strings);

    return {
      ...text_inputs.toJson(),
      ...?image_inputs?.toJson(),
    };
  }

  static Future<Processor> from_pretrained(
    String pretrained_model_name_or_path,
    [ProcessorPretrainedOptions? options]
  ) async => Processor.setup_from_pretrained(reflection)(pretrained_model_name_or_path, options);
}
