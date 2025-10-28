import 'dart:math' as math;

import 'package:transformers/src/base/image_processing_utils.dart';
import 'package:transformers/src/utils/image.dart';
import 'package:transformers/src/utils/tensor.dart';

class Idefics3ImageProcessor extends ImageProcessor {
  bool do_image_splitting;
  Map<String, dynamic> max_image_size;

  Idefics3ImageProcessor(super.config)
      : do_image_splitting = config['do_image_splitting'] ?? true,
        max_image_size = config['max_image_size'];

  /// Calculate size to resize images to, to be multiples of `vision_encoder_max_size` while preserving the aspect ratio.
  /// @param {Tensor} pixel_values Tensor of the image to resize.
  /// @param {number} vision_encoder_max_size Maximum size of the output image. If the image is larger than this size,
  /// it will be split into patches of this size, and the original image will be concatenated with the patches, resized to max_size.
  ({int height, int width}) get_resize_for_vision_encoder<T>(Tensor<T> pixel_values, int vision_encoder_max_size) {
    int height = pixel_values.dims[pixel_values.dims.length - 2], width = pixel_values.dims.last;

    final aspect_ratio = width / height;
    if (width >= height) {
      width = (width / vision_encoder_max_size).ceil() * vision_encoder_max_size;
      height = (width / aspect_ratio).floor();
      height = (height / vision_encoder_max_size).ceil() * vision_encoder_max_size;
    } else {
      height = (height / vision_encoder_max_size).ceil() * vision_encoder_max_size;
      width = (height * aspect_ratio).floor();
      width = (width / vision_encoder_max_size).ceil() * vision_encoder_max_size;
    }
    return (height: height, width: width);
  }

  /// @param {RawImage|RawImage[]|RawImage[][]} images
  @override
  Future<ImageProcessorResult> call(dynamic images, [List<dynamic> args = const []]) async {
    final [Map<String, dynamic> kwargs] = args;
    bool? do_image_splitting = kwargs['do_image_splitting'];
    bool return_row_col_info = kwargs['return_row_col_info'] ?? false;

    /** @type {RawImage[][]} */
    List<List<RawImage>> batched_2d_images;
    if (images is RawImage) {
      batched_2d_images = [[images]];
    } else {
      images = images as List<dynamic>;
      if (images.isEmpty || images.firstOrNull == null) {
        throw ArgumentError('No images provided.');
      }

      if (images.first is RawImage) {
        batched_2d_images = [images as List<RawImage>];
      } else if (images.first is List<RawImage>) {
        batched_2d_images = images as List<List<RawImage>>;
      } else {
        throw ArgumentError('Images must be of type RawImage, List<RawImage>, or List<List<RawImage>>.');
      }
    }

    // List of tensors, each with shape [patches, channels, height, width]
    final List<Tensor<double>> all_pixel_values = [];
    final List<List<int>>images_list_rows = [];
    final List<List<int>>images_list_cols = [];

    final List<(int, int)> original_sizes = [];
    final List<(int, int)> reshaped_input_sizes = [];
    for (final image_batch in batched_2d_images) {
      List<PreprocessedImage> images_list = await Future.wait(image_batch.map(preprocess));

      // Original sizes of images
      original_sizes.addAll(images_list.map((x) => x.original_size));

      // Reshaped sizes of images, before padding or cropping
      reshaped_input_sizes.addAll(images_list.map((x) => x.reshaped_input_size));

      // Convert images to 4D tensors for easier processing
      await Future.wait(images_list.map((x) async {
        x.pixel_values = await x.pixel_values.unsqueeze(0);
        return x;
      }));

      final int longest_edge = this.max_image_size['longest_edge'];

      List<Tensor<double>> images_tensor;
      if (do_image_splitting ?? this.do_image_splitting) {
        final List<int> image_rows = List.filled(images_list.length, 0);
        final List<int> image_cols = List.filled(images_list.length, 0);

        // We first resize both height and width of each image to the nearest max_image_size multiple, disregarding the aspect ratio
        images_tensor = await Future.wait(images_list.indexed.map((e) async {
          final (i, x) = e;
          final new_size = get_resize_for_vision_encoder(x.pixel_values, longest_edge);

          final resized = await interpolate_4d(
            x.pixel_values,
            size: [new_size.height, new_size.width],
          );

          final (frames, num_splits_h, num_splits_w) = await split_image(
            resized,
            longest_edge: longest_edge,
          );
          image_rows[i] = num_splits_h;
          image_cols[i] = num_splits_w;
          return cat(frames.map((t) => t.cast<double>()).toList(), 0);
        }));

        images_list_rows.add(image_rows);
        images_list_cols.add(image_cols);
      } else {
        final List<int> size = [longest_edge, longest_edge];
        images_tensor = await Future.wait(
          images_list.map((x) async => (await interpolate_4d(x.pixel_values, size: size)).cast<double>()),
        );

        images_list_rows.add(List.filled(images_list.length, 0));
        images_list_cols.add(List.filled(images_list.length, 0));
      }

      all_pixel_values.add(await cat(images_tensor, 0));
    }

    final int batch_size = all_pixel_values.length;
    final [n, c, h, w] = all_pixel_values[0].dims;

    // Stack pixel values
    Tensor<double> pixel_values;
    Tensor<bool> pixel_attention_mask;
    if (batch_size == 1) {
      all_pixel_values[0] = await all_pixel_values[0].unsqueeze(0);
      pixel_values = all_pixel_values[0];
      pixel_attention_mask = await full([batch_size, n, h, w], true);
    } else {
      // Add padding (if necessary) to images with less patches than the maximum number of patches
      final max_num_patches = all_pixel_values.map((x) => x.dims.first).reduce(math.max);

      pixel_attention_mask = await full([batch_size, max_num_patches, h, w], true);
      final pixel_attention_mask_data = pixel_attention_mask.data;
      final pixel_attention_mask_stride = max_num_patches * h * w;
      for (int i=0; i < batch_size; ++i) {
        final num_patches = all_pixel_values[i].dims[0];
        if (num_patches < max_num_patches) {
          all_pixel_values[i] = await cat([
            all_pixel_values[i],
            await full([max_num_patches - num_patches, c, h, w], 0),
          ], 0);

          final start_offset = i * pixel_attention_mask_stride + num_patches * h * w;
          final end_offset = (i + 1) * pixel_attention_mask_stride;

          for (int j=start_offset; j < end_offset; j++) {
            pixel_attention_mask_data[j] = false;
          }
        }
      }
      pixel_values = await stack(all_pixel_values, 0);
    }

    return ImageProcessorResult(
      pixel_values: pixel_values,
      pixel_attention_mask: pixel_attention_mask,

      original_sizes: original_sizes,
      reshaped_input_sizes: reshaped_input_sizes,

      rows: return_row_col_info ? images_list_rows : null,
      cols: return_row_col_info ? images_list_cols : null,
    );
  }

  Future<(List<Tensor>, int, int)> split_image(Tensor pixel_values, { required int longest_edge }) async {
    final max_height = longest_edge;
    final max_width = longest_edge;

    final List<Tensor> frames = [];

    int height = pixel_values.dims[pixel_values.dims.length - 2],
        width = pixel_values.dims.last;

    int num_splits_h = 0, num_splits_w = 0;

    if (height > max_height || width > max_width) {
      // Calculate the number of splits
      num_splits_h = (height / max_height).ceil();
      num_splits_w = (width / max_width).ceil();

      // Calculate the optimal width and height for the sub-images
      final optimal_height = (height / num_splits_h).ceil();
      final optimal_width = (width / num_splits_w).ceil();

      // Iterate through each row and column
      for (int r = 0; r < num_splits_h; ++r) {
        for (int c = 0; c < num_splits_w; ++c) {
          int start_x, start_y, end_x, end_y;
          if (r == num_splits_h - 1) { // At bottom
            start_y = height - optimal_height;
            end_y = height;
          } else {
            start_y = r * optimal_height;
            end_y = (r + 1) * optimal_height;
          }
          if (c == num_splits_w - 1) { // At right
            start_x = width - optimal_width;
            end_x = width;
          } else {
            start_x = c * optimal_width;
            end_x = (c + 1) * optimal_width;
          }

          final starts = [start_y, start_x];
          final ends = [end_y, end_x];

          final patch = await slice(pixel_values, starts, ends, [2, 3]);
          frames.add(patch);
        }
      }

      // Resize the global image to match max dimensions for memory efficiency
      final global_image_height = max_height;
      final global_image_width = max_width;

      if (height != global_image_height || width != global_image_width) {
        pixel_values = await interpolate_4d(
          pixel_values,
          size: [global_image_height, global_image_width],
        );
      }
    }

    frames.add(pixel_values);

    return (frames, num_splits_h, num_splits_w);
  }
}
