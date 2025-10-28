import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as path;
import 'package:transformers/src/utils/image.dart';

Future<void> checkFileHash(String filePath, String expectedHash) async {
  final hash = await sha256.bind(File(filePath).openRead()).first;
  expect(hash.toString(), expectedHash);
}

void main() {
  group('load_image', () {
    test('can load a huggingface image', () async {
      final image = await load_image('https://huggingface.co/ibm-granite/granite-docling-258M/resolve/main/assets/new_arxiv.png');
      final outPath = path.join('test', 'utils', 'test_artifacts', 'out', 'new_arxiv.png');

      await image.save(outPath);

      await checkFileHash(outPath, 'c54ec9d21a0ae271e384bf767cc26f933d01d99851afaa0cfc13c0985e46db5a');
    });

    test('can load a local image', () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish.gif');

      final image = await load_image(inPath);

      final hash = sha256.convert(image.toUint8List()).toString();
      expect(hash, '366f8481103ec5526d71541726402f00e732a39b8ce4497cf4924db382db3ac3');
    });
  });

  group('RawImage', () {
    test('can make an image grayscale', () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish.gif');
      final outPath = path.join('test', 'utils', 'test_artifacts', 'out', 'fish_grayscale.png');

      final image = await load_image(inPath);
      await image
          .grayscale()
          .save(outPath);

      await checkFileHash(outPath, '21a1afd4dea01c61d25cfdf8f3c9577f201d9d5b121208c22d05ab87cfe004f4');
    });

    test('can make an image rgb', () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish_grayscale.png');
      final outPath = path.join('test', 'utils', 'test_artifacts', 'out', 'fish_grayscale_to_rgb.png');

      final image = await load_image(inPath);
      await image
          .rgb()
          .save(outPath);

      await checkFileHash(outPath, '5b824e2c60b6afd87eee39c691e3c0813ebe2bdeed4176d0d43e540bcd4d1dc5');
    });

    test('can make an alpha mask', () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish.gif');
      final maskPath = path.join('test', 'utils', 'test_artifacts', 'fish_alpha_mask.png');
      final outPath = path.join('test', 'utils', 'test_artifacts', 'out', 'fish_alpha_mask.png');

      final mask = await load_image(maskPath);
      final image = await load_image(inPath);
      await image
        .putAlpha(mask)
        .save(outPath);

      await checkFileHash(outPath, '8ebe25805fad7494977afa7ea922167285aab48d9e99ed91ce50509afb35e0bf');
    });

    test('can resize an image', () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish.gif');
      final outPath = path.join('test', 'utils', 'test_artifacts', 'out', 'fish_resized.png');

      final image = await load_image(inPath);
      await image
          .resize(25, -1)
          .save(outPath);

      await checkFileHash(outPath, 'cb073c48f5f7d81f8e7971f7bfb926b39a97d5ea2331b18113c06e2949a3adef');
    });

    test('can pad an image', () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish.gif');
      final outPath = path.join('test', 'utils', 'test_artifacts', 'out', 'fish_padded.png');

      final image = await load_image(inPath);
      await image
          .pad(25, 50, 75, 100)
          .save(outPath);

      await checkFileHash(outPath, 'ef5d7c13a0247b5ee6aec37fd78accbc94139eb5977613821cc1bb56801c148e');
    });

    test('can crop an image', () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish.gif');
      final outPath = path.join('test', 'utils', 'test_artifacts', 'out', 'fish_cropped.png');

      final image = await load_image(inPath);
      await image
          .crop(25, 50, 75, 100)
          .save(outPath);

      await checkFileHash(outPath, '93868b96bcf7f9a709a351614f6b98a39e81536b3437de96980add89338f6243');
    });

    test('can center crop an image', () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish.gif');
      final outPath = path.join('test', 'utils', 'test_artifacts', 'out', 'fish_center_cropped.png');

      final image = await load_image(inPath);
      await image
          .center_crop(100, 100)
          .save(outPath);

      await checkFileHash(outPath, '351227ba68eaf18a97baacc71b48ef0a8859497e71b875dab2aec306bd0567f2');
    });

    test("can split an image into it's different channels", () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish.gif');
      final outPath = path.join('test', 'utils', 'test_artifacts', 'out');
      final List<(String, String)> expectedOutputs = [
        (path.join(outPath, 'fish_r.png'), '729fe4ed6ce3f2b073a7954e7f69cae4c1fef79acd109766ef3e3e48f271b114'),
        (path.join(outPath, 'fish_g.png'), '1a9a63a691b0d551e2054c017bba06d468c56c4bf0cd9f5408113dc3755f2909'),
        (path.join(outPath, 'fish_b.png'), 'd96c81716603274025b5a4579c641c97381b4995d3c5a64deaa07520aaa82032'),
      ];

      final image = await load_image(inPath);
      final splits = image.split();
      expect(splits.length, 3);

      for (int i=0; i < splits.length; i++) {
        final (outPath, hash) = expectedOutputs[i];
        await splits[i].save(outPath);
        await checkFileHash(outPath, hash);
      }
    });

    test("can properly clone an image", () async {
      final inPath = path.join('test', 'utils', 'test_artifacts', 'fish.gif');
      final outPathOriginal = path.join('test', 'utils', 'test_artifacts', 'out', 'fish_original_clone.png');
      final outPathClone = path.join('test', 'utils', 'test_artifacts', 'out', 'fish_clone.png');

      final image = await load_image(inPath);
      final clone = image.clone().grayscale();
      image.resize(10, 10);

      await image.save(outPathOriginal);
      await clone.save(outPathClone);

      await checkFileHash(outPathOriginal, '428700cda7a82bd8099eabc7d376787f4cb97e0da46b389f28efe6dfb87cdee2');
      await checkFileHash(outPathClone, '21a1afd4dea01c61d25cfdf8f3c9577f201d9d5b121208c22d05ab87cfe004f4');

      final hashOriginal = await sha256.bind(File(outPathOriginal).openRead()).first;
      final hashClone = await sha256.bind(File(outPathClone).openRead()).first;
      expect(hashOriginal, isNot(hashClone));
    });
  });
}
