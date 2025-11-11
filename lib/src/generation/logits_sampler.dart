import 'dart:math' as math;

import 'package:transformers/src/generation/configuration_utils.dart';
import 'package:transformers/src/utils/maths.dart';
import 'package:transformers/src/utils/tensor.dart';

final _rand = math.Random();

/// Sampler is a base class for all sampling methods used for text generation.
class LogitsSampler {
  final GenerationConfig generation_config;

  /// Creates a new Sampler object with the specified generation config.
  /// @param {GenerationConfig} generation_config The generation config.
  LogitsSampler(this.generation_config);

  /// Executes the sampler, using the specified logits.
  /// @param {Tensor} logits
  /// @returns {Promise<[bigint, number][]>}
  Future<List<(int, double)>> call(Tensor logits) async {
    // Sample from logits, of dims [batch, sequence_length, vocab_size].
    // If index is specified, sample from [batch, index, vocab_size].
    return sample(logits);
  }

  /// Abstract method for sampling the logits.
  /// @param {Tensor} logits
  /// @throws {Error} If not implemented in subclass.
  /// @returns {Promise<[bigint, number][]>}
  Future<List<(int, double)>> sample(Tensor logits) async {
    throw UnimplementedError('sample should be implemented in subclasses.');
  }

  /// Returns the specified logits as an array, with temperature applied.
  /// @param {Tensor} logits
  /// @param {number} index
  /// @returns {Float32Array}
  List<int> getLogits(Tensor logits, int index) {
    int vocabSize = logits.dims.last;

    List<int> logs = logits.data as List<int>;

    if (index == -1) {
      logs = logs.sublist(logs.length - vocabSize);
    } else {
      int startIndex = index * vocabSize;
      logs = logs.sublist(startIndex, startIndex + vocabSize);
    }
    return logs;
  }

  /// Selects an item randomly based on the specified probabilities.
  /// @param {import("../transformers.js").DataArray} probabilities An array of probabilities to use for selection.
  /// @returns {number} The index of the selected item.
  int randomSelect(List<num> probabilities) {
    // Return index of chosen item
    num sumProbabilities = 0;
    for (int i = 0; i < probabilities.length; ++i) {
      sumProbabilities += probabilities[i];
    }

    double r = _rand.nextDouble() * sumProbabilities;
    for (int i = 0; i < probabilities.length; ++i) {
      r -= probabilities[i];
      if (r <= 0) {
        return i;
      }
    }

    return 0; // return first (most probable) as a fallback
  }

  /// Returns a Sampler object based on the specified options.
  /// @param {GenerationConfig} generation_config An object containing options for the sampler.
  /// @returns {LogitsSampler} A Sampler object.
  static LogitsSampler getSampler(GenerationConfig generation_config) {
    // - *greedy decoding*: `num_beams=1` and `do_sample=False`
    // - *contrastive search*: `penalty_alpha>0` and `top_k>1`
    // - *multinomial sampling*: `num_beams=1` and `do_sample=True`
    // - *beam-search decoding*: `num_beams>1` and `do_sample=False`
    // - *beam-search multinomial sampling*: `num_beams>1` and `do_sample=True`
    // - *diverse beam-search decoding*: `num_beams>1` and `num_beam_groups>1`
    // - *constrained beam-search decoding*: `constraints!=None` or `force_words_ids!=None`

    // NOTE: beam search is implemented directly into the generation function
    if (generation_config.do_sample) {
      return MultinomialSampler(generation_config);

    } else if (generation_config.num_beams > 1) {
      return BeamSearchSampler(generation_config);

    } else {
      final num_return_sequences = generation_config.num_return_sequences;
      if (num_return_sequences != null && num_return_sequences > 1) {
        throw Exception('num_return_sequences has to be 1 when doing greedy search, but is $num_return_sequences.');
      }
      return GreedySampler(generation_config);
    }
  }
}

/// Class representing a Greedy Sampler.
class GreedySampler extends LogitsSampler {
  GreedySampler(super.generation_config);

  /// Sample the maximum probability of a given logits tensor.
  /// @param {Tensor} logits
  /// @returns {Promise<[bigint, number][]>} An array with a single tuple, containing the index of the maximum value and a meaningless score (since this is a greedy search).
  @override
  Future<List<(int, double)>> sample(Tensor logits) async {
    // NOTE: no need to do log_softmax here since we only take the maximum
    final argmax = max(List<num>.from(logits.data)).$2;

    // Note: score is meaningless in this context, since we are performing
    // greedy search (p = 1 => log(p) = 0)
    return [
      (argmax, 0),
    ];
  }
}

/// Class representing a MultinomialSampler.
class MultinomialSampler extends LogitsSampler {
  MultinomialSampler(super.generation_config);

  /// Sample from the logits.
  /// @param {Tensor} logits
  /// @returns {Promise<[bigint, number][]>}
  @override
  Future<List<(int, double)>> sample(Tensor logits) async {
    int k = logits.dims.last; // defaults to vocab size
    if (generation_config.top_k > 0) {
      k = math.min(generation_config.top_k, k);
    }

    // Get top k tokens
    final (v, i) = await topk(logits, k);

    // Compute softmax over logits
    final probabilities = softmax(List<num>.from(v.data));

    return List.generate(generation_config.num_beams, (_) {
      final sampledIndex = randomSelect(probabilities);
      return (
        i.data[sampledIndex], // token id
        math.log(probabilities[sampledIndex]), // score
      );
    });
  }
}

/// Class representing a BeamSearchSampler.
class BeamSearchSampler extends LogitsSampler {
  BeamSearchSampler(super.generation_config);

  /// Sample from the logits.
  /// @param {Tensor} logits
  /// @returns {Promise<[bigint, number][]>}
  @override
  Future<List<(int, double)>> sample(Tensor logits) async {
    int k = logits.dims.last; // defaults to vocab size
    if (generation_config.top_k > 0) {
      k = math.min(generation_config.top_k, k);
    }

    // Get top k tokens
    final (v, i) = await topk(logits, k);

    // Compute softmax over logits
    final probabilities = softmax(v.data as List<num>);

    return List.generate(generation_config.num_beams, (x) => (
      i.data[x], // token id
      math.log(probabilities[x]), // score
    ));
  }
}
