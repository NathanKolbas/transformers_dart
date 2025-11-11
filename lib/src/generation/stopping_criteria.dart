// NOTE:
// Stopping Criteria returns a list of `batch_size` booleans, indicating whether each sequence in the batch should be stopped.

import 'dart:collection';

/// Abstract base class for all stopping criteria that can be applied during generation.
class StoppingCriteria {
  /// @param {number[][]} input_ids (`number[][]` of shape `(batch_size, sequence_length)`):
  /// Indices of input sequence tokens in the vocabulary.
  /// @param {number[][]} scores scores (`number[][]` of shape `(batch_size, config.vocab_size)`):
  /// Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
  /// or scores for each vocabulary token after SoftMax.
  /// @returns {boolean[]} A list of booleans indicating whether each sequence should be stopped.
  List<bool> call(List<dynamic> input_ids, [List<List<int>>? scores]) {
    throw UnimplementedError('StoppingCriteria needs to be subclassed');
  }
}

class StoppingCriteriaList with IterableMixin<StoppingCriteria> {
  final List<StoppingCriteria> criteria;

  /// Constructs a new instance of `StoppingCriteriaList`.
  StoppingCriteriaList([List<StoppingCriteria>? criteria]) : criteria = criteria ?? [];

  /// Adds a new stopping criterion to the list.
  ///
  /// @param {StoppingCriteria} item The stopping criterion to add.
  void push(StoppingCriteria item) {
    criteria.add(item);
  }

  /// Adds multiple stopping criteria to the list.
  ///
  /// @param {StoppingCriteria|StoppingCriteriaList|StoppingCriteria[]} items The stopping criteria to add.
  void extend(dynamic items) {
    if (items is StoppingCriteriaList) {
      items = items.criteria;
    } else if (items is StoppingCriteria) {
      items = [items];
    }
    criteria.addAll(items);
  }

  List<bool> call(List<dynamic> input_ids, [List<List<int>>? scores]) {
    final List<bool> is_done = List.filled(input_ids.length, false);
    for (final criterion in criteria) {
      final criterion_done = criterion(input_ids, scores);
      for (int i = 0; i < is_done.length; ++i) {
        is_done[i] = is_done[i] || criterion_done[i];
      }
    }
    return is_done;
  }

  @override
  Iterator<StoppingCriteria> get iterator => criteria.iterator;
}

/// This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`.
/// Keep in mind for decoder-only type of transformers, this will include the initial prompted tokens.
class MaxLengthCriteria extends StoppingCriteria {
  final int max_length;
  final int? max_position_embeddings;

  ///
  /// @param {number} max_length The maximum length that the output sequence can have in number of tokens.
  /// @param {number} [max_position_embeddings=null] The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
  MaxLengthCriteria(this.max_length, [this.max_position_embeddings]);

  @override
  List<bool> call(List<dynamic> input_ids, [List<List<int>>? scores]) {
    return input_ids.map((ids) => (ids as List).length >= max_length).toList();
  }
}

/// This class can be used to stop generation whenever the "end-of-sequence" token is generated.
/// By default, it uses the `model.generation_config.eos_token_id`.
class EosTokenCriteria extends StoppingCriteria {
  final List<int> eos_token_id;

  ///
  /// @param {number|number[]} eos_token_id The id of the *end-of-sequence* token.
  /// Optionally, use a list to set multiple *end-of-sequence* tokens.
  EosTokenCriteria(dynamic eos_token_id)
      : eos_token_id = List<int>.from(eos_token_id is List ? eos_token_id : [eos_token_id]);

  ///
  /// @param {number[][]} input_ids
  /// @param {number[][]} scores
  /// @returns {boolean[]}
  @override
  List<bool> call(List<dynamic> input_ids, [List<List<int>>? scores]) {
    return input_ids.map((ids) {
      final last = ids.last;
      // NOTE: We use == instead of === to allow for number/bigint comparison
      return eos_token_id.any((eos_id) => last == eos_id);
    }).toList();
  }
}
