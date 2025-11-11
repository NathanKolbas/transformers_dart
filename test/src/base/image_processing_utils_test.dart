import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as path;
import 'package:transformers/src/base/image_processing_utils.dart';
import 'package:transformers/src/utils/image.dart';

Future<String> getFileHash(String filePath) async {
  return (await sha256.bind(File(filePath).openRead()).first).toString();
}

const Map<String, dynamic> imageProcessorConfig = {
  'image_mean': null,
  'mean': null, // Otherwise
  'image_std': null,
  'std': null, // Otherwise
  'resample': null,
  'do_rescale': null,
  'rescale_factor': null,
  'do_normalize': false,
  'do_thumbnail': false,
  'size': null,
  'image_size': null, // Otherwise
  'do_resize': null,
  'size_divisibility': null,
  'size_divisor': null, // Otherwise
  'do_center_crop': null,
  'crop_size': null,
  'do_convert_rgb': null,
  'do_crop_margin': false,
  'pad_size': null,
  'do_pad': false,
  'min_pixels': null,
  'max_pixels': null,
  'do_flip_channel_order': null,
};

void main() {
  group('ImageProcessor', () {
    test('can crop the margin of an image', () async {
      final imageProcessor = ImageProcessor(imageProcessorConfig);

      final inPath = path.join('test', 'base', 'test_artifacts', 'mike.jpg');
      final outPath = path.join('test', 'base', 'test_artifacts', 'out', 'mike_crop_margin.jpg');

      final image = (await load_image(inPath)).image;
      final padded = img.Image(
        width: image.width + 50,
        height: image.height + 50,
        numChannels: image.numChannels,
      );
      img.fill(padded, color: img.ColorRgb8(255, 255, 255));
      img.compositeImage(padded, image, dstX: 25, dstY: 25);
      final rawImage = RawImage.fromImage(image);
      await imageProcessor.crop_margin(rawImage).save(outPath);

      final hash = await getFileHash(outPath);
      expect(hash, 'f3e9a584f547df9fab09154c526ce260b6b028790e22bc9c60e00f22e35a5e13');
    });

    // TODO: Write tests for the other method but later once we confirm this is actually the implementation we want to use
    // test('can pad an image', () async {
    //   final imageProcessor = ImageProcessor(imageProcessorConfig);
    //
    //   final inPath = path.join('test', 'base', 'test_artifacts', 'mike.jpg');
    //   final outPath = path.join('test', 'base', 'test_artifacts', 'out', 'mike_padded.jpg');
    //
    //   final image = await load_image(inPath);
    //   final (pixelData, [width, height, channels]) = imageProcessor.pad_image(
    //     image.data.toList().cast<double>(),
    //     [image.width, image.height, image.channels],
    //     50,
    //   );
    //   final rawImage = RawImage.fromImage(
    //     img.Image(width: width, height: height, numChannels: channels)
    //     ..data = img.ImageData,
    //   );
    //
    //   final hash = await getFileHash(outPath);
    //   expect(hash, 'f3e9a584f547df9fab09154c526ce260b6b028790e22bc9c60e00f22e35a5e13');
    // });
  });
}
