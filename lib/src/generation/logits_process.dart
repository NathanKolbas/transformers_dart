import 'dart:collection';
import 'dart:convert';
import 'dart:math' as math;

import 'package:transformers/src/utils/tensor.dart';

abstract class LogitsBase {
  /// Apply the processor to the input logits.
  ///
  /// @abstract
  /// @param {bigint[][]} input_ids The input ids.
  /// @param {Tensor} logits The logits to process.
  /// @throws {Error} Throws an error if `_call` is not implemented in the subclass.
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    throw UnimplementedError("`_call` should be implemented in a subclass");
  }
}

/// Abstract base class for all logit processors that can be applied during generation.
class LogitsProcessor extends LogitsBase {}

/// Abstract base class for all logit warpers that can be applied during generation with multinomial sampling.
class LogitsWarper extends LogitsBase {}

/// A class representing a list of logits processors. A logits processor is a function that modifies the logits
/// output of a language model. This class provides methods for adding new processors and applying all processors to a
/// batch of logits.
class LogitsProcessorList with IterableMixin<LogitsBase> {
  final List<LogitsBase> processors = [];

  /// Adds a new logits processor to the list.
  ///
  /// @param {LogitsProcessor} item The logits processor function to add.
  void add(LogitsBase item) {
    processors.add(item);
  }

  /// Adds multiple logits processors to the list.
  ///
  /// @param {LogitsProcessor[]} items The logits processor functions to add.
  void extend(List<LogitsBase> items) {
    processors.addAll(items);
  }

  /// Applies all logits processors in the list to a batch of logits, modifying them in-place.
  ///
  /// @param {bigint[][]} input_ids The input IDs for the language model.
  /// @param {Tensor} logits
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    Tensor toReturn = logits;
    // NOTE: Most processors modify logits inplace
    for (final processor in processors) {
      toReturn = await processor(input_ids, toReturn);
    }
    return toReturn;
  }

  @override
  Iterator<LogitsBase> get iterator => processors.iterator;
}

/// A LogitsProcessor that forces a BOS token at the beginning of the generated sequence.
class ForcedBOSTokenLogitsProcessor extends LogitsProcessor {
  final int bos_token_id;

  /// Create a ForcedBOSTokenLogitsProcessor.
  /// @param {number} bos_token_id The ID of the beginning-of-sequence token to be forced.
  ForcedBOSTokenLogitsProcessor(this.bos_token_id);

  /// Apply the BOS token forcing to the logits.
  /// @param {bigint[][]} input_ids The input IDs.
  /// @param {Tensor} logits The logits.
  /// @returns {Tensor} The logits with BOS token forcing.
  /// TODO: Validate that the changes work correctly due to no updating list in-place
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    final List<Future<void>> disposers = [];
    final List<dynamic> updatedData = [];

    for (int i = 0; i < input_ids.length; ++i) {
      final batch_logits = await logits[i];
      final batch_logits_data = batch_logits.data;

      if (input_ids[i].length == 1) {
        batch_logits_data.fillRange(0, batch_logits_data.length, -double.infinity);
        batch_logits_data[bos_token_id] = 0;
      }

      updatedData.addAll(batch_logits_data);
      disposers.add(batch_logits.dispose());
    }

    await Future.wait([
      ...disposers,
      logits.updateData(updatedData),
    ]);
    return logits;
  }
}

/// A logits processor that enforces the specified token as the last generated token when `max_length` is reached.
class ForcedEOSTokenLogitsProcessor extends LogitsProcessor {
  final int max_length;
  final List<int> eos_token_id;

  /// Create a ForcedEOSTokenLogitsProcessor.
  /// @param {number} max_length The maximum length of the sequence to be generated.
  /// @param {number|number[]} eos_token_id The id(s) of the *end-of-sequence* token.
  ForcedEOSTokenLogitsProcessor(
    this.max_length,
    dynamic eos_token_id,
  ) : eos_token_id = (eos_token_id is List ? eos_token_id : [eos_token_id]) as List<int>;

  /// Apply the processor to input_ids and logits.
  ///
  /// @param {bigint[][]} input_ids The input ids.
  /// @param {Tensor} logits The logits tensor.
  /// TODO: Validate that the changes work correctly due to no updating list in-place
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    final List<Future<void>> disposers = [];
    final List<dynamic> updatedData = [];

    for (int i = 0; i < input_ids.length; ++i) {
      final batch_logits = await logits[i];
      final batch_logits_data = batch_logits.data;

      if (input_ids[i].length == max_length - 1) {
        batch_logits_data.fillRange(0, batch_logits_data.length, -double.infinity);
        for (final eos_token in eos_token_id) {
          batch_logits_data[eos_token] = 0;
        }
      }

      updatedData.addAll(batch_logits_data);
      disposers.add(batch_logits.dispose());
    }

    await Future.wait([
      ...disposers,
      logits.updateData(updatedData),
    ]);
    return logits;
  }
}

/// A LogitsProcessor that suppresses a list of tokens as soon as the `generate` function starts
/// generating using `begin_index` tokens. This should ensure that the tokens defined by
/// `begin_suppress_tokens` at not sampled at the begining of the generation.
class SuppressTokensAtBeginLogitsProcessor extends LogitsProcessor {
  final List<int> begin_suppress_tokens;
  final int begin_index;

  /// Create a SuppressTokensAtBeginLogitsProcessor.
  /// @param {number[]} begin_suppress_tokens The IDs of the tokens to suppress.
  /// @param {number} begin_index The number of tokens to generate before suppressing tokens.
  SuppressTokensAtBeginLogitsProcessor(this.begin_suppress_tokens, this.begin_index);

  /// Apply the BOS token forcing to the logits.
  /// @param {bigint[][]} input_ids The input IDs.
  /// @param {Tensor} logits The logits.
  /// @returns {Tensor} The logits with BOS token forcing.
  /// TODO: Validate that the changes work correctly due to no updating list in-place
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    final List<Future<void>> disposers = [];
    final List<dynamic> updatedData = [];

    for (int i = 0; i < input_ids.length; ++i) {
      final batch_logits = await logits[i];
      final batch_logits_data = batch_logits.data;

      if (input_ids[i].length == begin_index) {
        for (final token_id in begin_suppress_tokens) {
          batch_logits_data[token_id] = -double.infinity;
        }
      }

      updatedData.addAll(batch_logits_data);
      disposers.add(batch_logits.dispose());
    }

    await Future.wait([
      ...disposers,
      logits.updateData(updatedData),
    ]);
    return logits;
  }
}

/// A logits processor that disallows ngrams of a certain size to be repeated.
class NoRepeatNGramLogitsProcessor extends LogitsProcessor {
  /// The no-repeat-ngram size. All ngrams of this size can only occur once.
  final int no_repeat_ngram_size;

  /// Create a NoRepeatNGramLogitsProcessor.
  /// @param {number} no_repeat_ngram_size The no-repeat-ngram size. All ngrams of this size can only occur once.
  NoRepeatNGramLogitsProcessor(this.no_repeat_ngram_size);

  /// Generate n-grams from a sequence of token ids.
  /// @param {bigint[]} prevInputIds List of previous input ids
  /// @returns {Map<string, number[]>} Map of generated n-grams
  Map<String, List<int>> getNgrams(List<int> prevInputIds) {
    final curLen = prevInputIds.length;

    final List<List<int>> ngrams = [];
    for (int j = 0; j < curLen + 1 - no_repeat_ngram_size; ++j) {
      final List<int> ngram = [];
      for (int k = 0; k < no_repeat_ngram_size; ++k) {
        ngram.add(prevInputIds[j + k]);
      }
      ngrams.add(ngram);
    }

    final Map<String, List<int>> generatedNgram = {};
    for (final ngram in ngrams) {
      final prevNgram = ngram.sublist(0, ngram.length - 1);
      final prevNgramKey = jsonEncode(prevNgram);
      final prevNgramValue = generatedNgram[prevNgramKey] ?? [];
      prevNgramValue.add(ngram[ngram.length - 1]);
      generatedNgram[prevNgramKey] = prevNgramValue;
    }
    return generatedNgram;
  }

  /// Generate n-grams from a sequence of token ids.
  /// @param {Map<string, number[]>} bannedNgrams Map of banned n-grams
  /// @param {bigint[]} prevInputIds List of previous input ids
  /// @returns {number[]} Map of generated n-grams
  List<int> getGeneratedNgrams(Map<String, List<int>> bannedNgrams, List<int> prevInputIds) {
    final ngramIdx = prevInputIds.sublist(prevInputIds.length + 1 - no_repeat_ngram_size, prevInputIds.length);
    final banned = bannedNgrams[jsonEncode(ngramIdx)] ?? [];
    return banned;
  }

  /// Calculate banned n-gram tokens
  /// @param {bigint[]} prevInputIds List of previous input ids
  /// @returns {number[]} Map of generated n-grams
  List<int> calcBannedNgramTokens(List<int> prevInputIds) {
    final List<int> bannedTokens = [];
    if (prevInputIds.length + 1 < no_repeat_ngram_size) {
      // return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
      return bannedTokens;
    } else {
      final generatedNgrams = getNgrams(prevInputIds);
      final bannedTokens = getGeneratedNgrams(generatedNgrams, prevInputIds);
      return bannedTokens;
    }
  }

  /// Apply the no-repeat-ngram processor to the logits.
  /// @param {bigint[][]} input_ids The input IDs.
  /// @param {Tensor} logits The logits.
  /// @returns {Tensor} The logits with no-repeat-ngram processing.
  /// TODO: Validate that the changes work correctly due to no updating list in-place
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    final List<Future<void>> disposers = [];
    final List<dynamic> updatedData = [];

    for (int i = 0; i < input_ids.length; ++i) {
      final batch_logits = await logits[i];
      final batch_logits_data = batch_logits.data;
      final bannedTokens = calcBannedNgramTokens(input_ids[i]);

      for (final token in bannedTokens) {
        batch_logits_data[token] = -double.infinity;
      }

      updatedData.addAll(batch_logits_data);
      disposers.add(batch_logits.dispose());
    }

    await Future.wait([
      ...disposers,
      logits.updateData(updatedData),
    ]);
    return logits;
  }
}

/// A logits processor that prevents the repetition of previous tokens through a penalty.
/// This penalty is applied at most once per token. Note that, for decoder-only models like most LLMs,
/// the considered tokens include the prompt.
///
/// In the original [paper](https://huggingface.co/papers/1909.05858), the authors suggest the use of a
/// penalty of around 1.2 to achieve a good balance between truthful generation and lack of repetition.
/// To penalize and reduce repetition, use `penalty` values above 1.0, where a higher value penalizes
/// more strongly. To reward and encourage repetition, use `penalty` values between 0.0 and 1.0, where
/// a lower value rewards more strongly.
class RepetitionPenaltyLogitsProcessor extends LogitsProcessor {
  final double penalty;

  /// Create a RepetitionPenaltyLogitsProcessor.
  /// @param {number} penalty The parameter for repetition penalty.
  /// - 1.0 means no penalty. Above 1.0 penalizes previously generated tokens.
  /// - Between 0.0 and 1.0 rewards previously generated tokens.
  RepetitionPenaltyLogitsProcessor(this.penalty);

  /// Apply the repetition penalty to the logits.
  /// @param {bigint[][]} input_ids The input IDs.
  /// @param {Tensor} logits The logits.
  /// @returns {Tensor} The logits with repetition penalty processing.
  /// TODO: Validate that the changes work correctly due to no updating list in-place
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    final List<Future<void>> disposers = [];
    final List<dynamic> updatedData = [];

    for (int i = 0; i < input_ids.length; ++i) {
      final batch_logits = await logits[i];
      final batch_logits_data = batch_logits.data;

      for (final input_id in input_ids[i].toSet()) {
        final token = input_id;
        if (batch_logits_data[token] < 0) {
          batch_logits_data[token] *= penalty;
        } else {
          batch_logits_data[token] /= penalty;
        }
      }

      updatedData.addAll(batch_logits_data);
      disposers.add(batch_logits.dispose());
    }

    await Future.wait([
      ...disposers,
      logits.updateData(updatedData),
    ]);
    return logits;
  }
}

/// A logits processor that enforces a minimum number of tokens.
class MinLengthLogitsProcessor extends LogitsProcessor {
  final int min_length;
  final List<int> eos_token_id;

  /// Create a MinLengthLogitsProcessor.
  /// @param {number} min_length The minimum length below which the score of `eos_token_id` is set to negative infinity.
  /// @param {number|number[]} eos_token_id The ID/IDs of the end-of-sequence token.
  MinLengthLogitsProcessor(
    this.min_length,
    dynamic eos_token_id,
  ) : eos_token_id = (eos_token_id is List ? eos_token_id : [eos_token_id]) as List<int>;

  /// Apply logit processor.
  /// @param {bigint[][]} input_ids The input IDs.
  /// @param {Tensor} logits The logits.
  /// @returns {Tensor} The processed logits.
  /// TODO: Validate that the changes work correctly due to no updating list in-place
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    final List<Future<void>> disposers = [];
    final List<dynamic> updatedData = [];

    for (int i = 0; i < input_ids.length; ++i) {
      final batch_logits = await logits[i];
      final batch_logits_data = batch_logits.data;

      if (input_ids[i].length < min_length) {
        for (final eos_token in eos_token_id) {
          batch_logits_data[eos_token] = -double.infinity;
        }
      }

      updatedData.addAll(batch_logits_data);
      disposers.add(batch_logits.dispose());
    }

    await Future.wait([
      ...disposers,
      logits.updateData(updatedData),
    ]);
    return logits;
  }
}

/// A logits processor that enforces a minimum number of new tokens.
class MinNewTokensLengthLogitsProcessor extends LogitsProcessor {
  final int prompt_length_to_skip;
  final int min_new_tokens;
  final List<int> eos_token_id;

  /// Create a MinNewTokensLengthLogitsProcessor.
  /// @param {number} prompt_length_to_skip The input tokens length.
  /// @param {number} min_new_tokens The minimum *new* tokens length below which the score of `eos_token_id` is set to negative infinity.
  /// @param {number|number[]} eos_token_id The ID/IDs of the end-of-sequence token.
  MinNewTokensLengthLogitsProcessor(
    this.prompt_length_to_skip,
    this.min_new_tokens,
    dynamic eos_token_id,
  ) : eos_token_id = (eos_token_id is List ? eos_token_id : [eos_token_id]) as List<int>;

  /// Apply logit processor.
  /// @param {bigint[][]} input_ids The input IDs.
  /// @param {Tensor} logits The logits.
  /// TODO: Validate that the changes work correctly due to no updating list in-place
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    final List<Future<void>> disposers = [];
    final List<dynamic> updatedData = [];

    for (int i = 0; i < input_ids.length; ++i) {
      final batch_logits = await logits[i];
      final batch_logits_data = batch_logits.data;
      final new_tokens_length = input_ids[i].length - prompt_length_to_skip;

      if (new_tokens_length < min_new_tokens) {
        for (final eos_token in eos_token_id) {
          batch_logits_data[eos_token] = -double.infinity;
        }
      }

      updatedData.addAll(batch_logits_data);
      disposers.add(batch_logits.dispose());
    }

    await Future.wait([
      ...disposers,
      logits.updateData(updatedData),
    ]);
    return logits;
  }
}

class NoBadWordsLogitsProcessor extends LogitsProcessor {
  final List<List<int>> bad_words_ids;
  final List<int> eos_token_id;

  /// Create a `NoBadWordsLogitsProcessor`.
  /// @param {number[][]} bad_words_ids List of list of token ids that are not allowed to be generated.
  /// @param {number|number[]} eos_token_id The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
  NoBadWordsLogitsProcessor(
    this.bad_words_ids,
    dynamic eos_token_id,
  ) : eos_token_id = (eos_token_id is List ? eos_token_id : [eos_token_id]) as List<int>;

  /// Apply logit processor.
  /// @param {bigint[][]} input_ids The input IDs.
  /// @param {Tensor} logits The logits.
  /// @returns {Tensor} The processed logits.
  /// TODO: Validate that the changes work correctly due to no updating list in-place
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    final List<Future<void>> disposers = [];
    final List<dynamic> updatedData = [];

    for (int i = 0; i < input_ids.length; ++i) {
      final batch_logits = await logits[i];
      final batch_logits_data = batch_logits.data;
      final ids = input_ids[i];

      for (final bad_word_ids in bad_words_ids) {
        // There aren't enough tokens to match the banned sequence
        if (ids.length < bad_word_ids.length - 1) continue;

        // Whether to modify the logits of the last token in the bad word id sequence
        bool mark = true;

        // For each bad word in the list, if the current sequence of input ids ends with this sequence (excluding the last),
        // then we set the logits of the last bad word id to -Infinity.
        for (int j = 1; j <= bad_word_ids.length - 1; ++j) {
          if (bad_word_ids[-j - 1] != ids[-j]) {
            // We have found a mismatch
            mark = false;
            break;
          }
        }
        if (mark) {
          batch_logits_data[bad_word_ids.last] = -double.infinity;
        }
      }

      updatedData.addAll(batch_logits_data);
      disposers.add(batch_logits.dispose());
    }

    await Future.wait([
      ...disposers,
      logits.updateData(updatedData),
    ]);
    return logits;
  }
}

/// [`LogitsProcessor`] for classifier free guidance (CFG). The scores are split over the batch dimension,
/// where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
/// correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
/// weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.
///
/// See [the paper](https://huggingface.co/papers/2306.05284) for more information.
class ClassifierFreeGuidanceLogitsProcessor extends LogitsProcessor {
  final int guidance_scale;

  /// Create a `ClassifierFreeGuidanceLogitsProcessor`.
  /// @param {number} guidance_scale The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
  /// Higher guidance scale encourages the model to generate samples that are more closely linked to the input
  /// prompt, usually at the expense of poorer quality.
  ClassifierFreeGuidanceLogitsProcessor(this.guidance_scale) {
    if (guidance_scale <= 1) {
      throw ArgumentError(
        'Require guidance scale >1 to use the classifier free guidance processor, got guidance scale $guidance_scale.'
      );
    }
  }

  /// Apply logit processor.
  /// @param {bigint[][]} input_ids The input IDs.
  /// @param {Tensor} logits The logits.
  /// @returns {Tensor} The processed logits.
  /// TODO: Validate that the changes work correctly due to no updating list in-place
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    if (logits.dims[0] != 2 * input_ids.length) {
      throw Exception(
        'Logits should have twice the batch size of the input ids, the first half of batches corresponding to '
        'the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got '
        'batch size ${logits.dims[0]} for the logits and ${input_ids.length} for the input ids.'
      );
    }

    final unguided_bsz = input_ids.length;
    final cond_logits = await logits.slice([[0, unguided_bsz], null]);
    final uncond_logits = await logits.slice([[unguided_bsz, logits.dims[0]], null]);

    // Merge into uncond_logits (to save memory). This is equivalent to the following:
    // scores = uncond_logits + (cond_logits - uncond_logits) * guidance_scale
    for (int i = 0; i < uncond_logits.data.length; ++i) {
      uncond_logits.data[i] += (cond_logits.data[i] - uncond_logits.data[i]) * guidance_scale;
    }

    await Future.wait([
      cond_logits.dispose(),
      uncond_logits.updateData(uncond_logits.data),
    ]);
    return uncond_logits;
  }
}

/// [`LogitsWarper`] for temperature (exponential scaling output probability distribution), which effectively means
/// that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and [`TopKLogitsWarper`].
class TemperatureLogitsWarper extends LogitsWarper {
  double temperature;

  /// Create a `TemperatureLogitsWarper`.
  /// @param {number} temperature Strictly positive float value used to modulate the logits distribution.
  /// A value smaller than `1` decreases randomness (and vice versa), with `0` being equivalent to shifting
  /// all probability mass to the most likely token.
  TemperatureLogitsWarper(this.temperature) {
    if (temperature <= 0) {
      String errorMessage = '`temperature` (=$temperature) must be a strictly positive float, otherwise your next token scores will be invalid.';

      if (temperature == 0) {
        errorMessage += " If you're looking for greedy decoding strategies, set `do_sample=false`.";
      }

      // Below wasn't in transformers.js but I assume it should have been
      print(errorMessage);
    }
  }

  /// Apply logit warper.
  /// @param {bigint[][]} input_ids The input IDs.
  /// @param {Tensor} logits The logits.
  /// @returns {Tensor} The processed logits.
  @override
  Future<Tensor> call(List<dynamic> input_ids, Tensor logits) async {
    final batch_logits_data = logits.data;
    for (int i=0; i < batch_logits_data.length; ++i) {
      batch_logits_data[i] /= temperature;
    }
    return await logits.updateData(batch_logits_data);
  }
}

/// [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
/// Often used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].
class TopPLogitsWarper extends LogitsWarper {
  double top_p;
  double filter_value;
  int min_tokens_to_keep;

  /// Create a `TopPLogitsWarper`.
  /// @param {number} top_p If set to < 1, only the smallest set of most probable tokens with
  /// probabilities that add up to `top_p` or higher are kept for generation.
  /// @param {Object} options Additional options for the top-p sampling.
  /// @param {number} [options.filter_value=-Infinity] All filtered values will be set to this float value.
  /// @param {number} [options.min_tokens_to_keep=1] Minimum number of tokens that cannot be filtered.
  TopPLogitsWarper(this.top_p, {
    this.filter_value = -double.infinity,
    this.min_tokens_to_keep = 1,
  }) {
    if (top_p < 0 || top_p > 1.0) {
      throw ArgumentError('`top_p` must be a float > 0 and < 1, but is $top_p');
    }
    if (min_tokens_to_keep < 1) {
      throw ArgumentError('`min_tokens_to_keep` must be a positive integer, but is $min_tokens_to_keep');
    }
  }
}

/// [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.
/// Often used together with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].
class TopKLogitsWarper extends LogitsWarper {
  int top_k;
  double filter_value;

  /// Create a `TopKLogitsWarper`.
  /// @param {number} top_k If set to > 0, only the top `top_k` tokens are kept for generation.
  /// @param {Object} options Additional options for the top-k sampling.
  /// @param {number} [options.filter_value=-Infinity] All filtered values will be set to this float value.
  /// @param {number} [options.min_tokens_to_keep=1] Minimum number of tokens that cannot be filtered.
  TopKLogitsWarper(this.top_k, {
    this.filter_value = -double.infinity,
    int min_tokens_to_keep = 1,
  }) {
    if (top_k < 0) {
      throw ArgumentError('`top_k` must be a positive integer, but is $top_k');
    }

    top_k = math.max(top_k, min_tokens_to_keep);
  }
}
