import 'dart:convert';

import 'package:transformers/src/generation/streamers.dart';

/// Class that holds a configuration for a generation task.
class GenerationConfig {
  // Parameters that control the length of the output
  /// The maximum length the generated tokens can have.
  /// Corresponds to the length of the input prompt + `max_new_tokens`.
  /// Its effect is overridden by `max_new_tokens`, if also set.
  /// @type {number}
  /// @default 20
  int max_length;

  /// The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
  /// @type {number}
  /// @default null
  int? max_new_tokens;

  /// The minimum length of the sequence to be generated.
  /// Corresponds to the length of the input prompt + `min_new_tokens`.
  /// Its effect is overridden by `min_new_tokens`, if also set.
  /// @type {number}
  /// @default 0
  int min_length;

  /// The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
  /// @type {number}
  /// @default null
  int? min_new_tokens;

  /// Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
  /// - `true`, where the generation stops as soon as there are `num_beams` complete candidates;
  /// - `false`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates;
  /// - `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
  /// @type {boolean|"never"}
  /// @default false
  dynamic early_stopping;

  /// The maximum amount of time you allow the computation to run for in seconds.
  /// Generation will still finish the current pass after allocated time has been passed.
  /// @type {number}
  /// @default null
  int? max_time;

  // Parameters that control the generation strategy used
  /// Whether or not to use sampling; use greedy decoding otherwise.
  /// @type {boolean}
  /// @default false
  bool do_sample;

  /// Number of beams for beam search. 1 means no beam search.
  /// @type {number}
  /// @default 1
  int num_beams;

  /// Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
  /// See [this paper](https://huggingface.co/papers/1610.02424) for more details.
  /// @type {number}
  /// @default 1
  int num_beam_groups;

  /// The values balance the model confidence and the degeneration penalty in contrastive search decoding.
  /// @type {number}
  /// @default null
  int? penalty_alpha;

  /// Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
  /// @type {boolean}
  /// @default true
  bool use_cache;

  // Parameters for manipulation of the model output logits
  /// The value used to modulate the next token probabilities.
  /// @type {number}
  /// @default 1.0
  double temperature;

  /// The number of highest probability vocabulary tokens to keep for top-k-filtering.
  /// @type {number}
  /// @default 50
  int top_k;

  /// If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
  /// @type {number}
  /// @default 1.0
  double top_p;

  /// Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated.
  /// If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation.
  /// See [this paper](https://huggingface.co/papers/2202.00666) for more details.
  /// @type {number}
  /// @default 1.0
  double typical_p;

  /// If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled.
  /// In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model.
  /// See [Truncation Sampling as Language Model Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
  /// @type {number}
  /// @default 0.0
  double epsilon_cutoff;

  /// Eta sampling is a hybrid of locally typical sampling and epsilon sampling.
  /// If set to float strictly between 0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`.
  /// The latter term is intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model.
  /// See [Truncation Sampling as Language Model Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
  /// @type {number}
  /// @default 0.0
  double eta_cutoff;

  /// This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time.
  /// Note that `diversity_penalty` is only effective if `group beam search` is enabled.
  /// @type {number}
  /// @default 0.0
  double diversity_penalty;

  /// The parameter for repetition penalty. 1.0 means no penalty.
  /// See [this paper](https://huggingface.co/papers/1909.05858) for more details.
  /// @type {number}
  /// @default 1.0
  double repetition_penalty;

  /// The paramater for encoder_repetition_penalty.
  /// An exponential penalty on sequences that are not in the original input.
  /// 1.0 means no penalty.
  /// @type {number}
  /// @default 1.0
  double encoder_repetition_penalty;

  /// Exponential penalty to the length that is used with beam-based generation.
  /// It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
  /// Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.
  /// @type {number}
  /// @default 1.0
  double length_penalty;

  /// If set to int > 0, all ngrams of that size can only occur once.
  /// @type {number}
  /// @default 0
  int no_repeat_ngram_size;

  /// List of token ids that are not allowed to be generated.
  /// In order to get the token ids of the words that should not appear in the generated text, use
  /// `tokenizer(bad_words, { add_prefix_space: true, add_special_tokens: false }).input_ids`.
  /// @type {number[][]}
  /// @default null
  List<List<int>>? bad_words_ids;

  /// List of token ids that must be generated.
  /// If given a `number[][]`, this is treated as a simple list of words that must be included, the opposite to `bad_words_ids`.
  /// If given `number[][][]`, this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one can allow different forms of each word.
  /// @type {number[][]|number[][][]}
  /// @default null
  dynamic force_words_ids;

  /// Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones).
  /// It's highly recommended to set this flag to `true` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization.
  /// @type {boolean}
  /// @default false
  bool renormalize_logits;

  /// Custom constraints that can be added to the generation to ensure that the output will contain the use of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
  /// @type {Object[]}
  /// @default null
  dynamic constraints;

  /// The id of the token to force as the first generated token after the `decoder_start_token_id`.
  /// Useful for multilingual models like mBART where the first generated token needs to be the target language token.
  /// @type {number}
  /// @default null
  int? forced_bos_token_id;

  /// The id of the token to force as the last generated token when `max_length` is reached.
  /// Optionally, use a list to set multiple *end-of-sequence* tokens.
  /// @type {number|number[]}
  /// @default null
  dynamic forced_eos_token_id;

  /// Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash. Note that using `remove_invalid_values` can slow down generation.
  /// @type {boolean}
  bool remove_invalid_values;

  /// This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated.
  /// The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay.
  /// @type {[number, number]}
  /// @default null
  (int, int)? exponential_decay_length_penalty;

  /// A list of tokens that will be suppressed at generation.
  /// The `SuppressTokens` logit processor will set their log probs to `-inf` so that they are not sampled.
  /// @type {number[]}
  /// @default null
  List<int>? suppress_tokens;

  /// A streamer that will be used to stream the generation.
  /// @type {import('./streamers.js').TextStreamer}
  /// @default null
  TextStreamer? streamer;

  /// A list of tokens that will be suppressed at the beginning of the generation.
  /// The `SuppressBeginTokens` logit processor will set their log probs to `-inf` so that they are not sampled.
  /// @type {number[]}
  /// @default null
  List<int>? begin_suppress_tokens;

  /// A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling.
  /// For example, `[[1, 123]]` means the second generated token will always be a token of index 123.
  /// @type {[number, number][]}
  /// @default null
  List<(int, int)>? forced_decoder_ids;

  /// The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
  /// Higher guidance scale encourages the model to generate samples that are more closely linked to the input
  /// prompt, usually at the expense of poorer quality.
  /// @type {number}
  /// @default null
  int? guidance_scale;

  // Parameters that define the output variables of `generate`
  /// The number of independently computed returned sequences for each element in the batch.
  /// @type {number}
  /// @default 1
  int num_return_sequences;

  /// Whether or not to return the attentions tensors of all attention layers.
  /// See `attentions` under returned tensors for more details.
  /// @type {boolean}
  /// @default false
  bool output_attentions;

  /// Whether or not to return the hidden states of all layers.
  /// See `hidden_states` under returned tensors for more details.
  /// @type {boolean}
  /// @default false
  bool output_hidden_states;

  /// Whether or not to return the prediction scores.
  /// See `scores` under returned tensors for more details.
  /// @type {boolean}
  /// @default false
  bool output_scores;

  /// Whether or not to return a `ModelOutput` instead of a plain tuple.
  /// @type {boolean}
  /// @default false
  bool return_dict_in_generate;

  // Special tokens that can be used at generation time
  /// The id of the *padding* token.
  /// @type {number}
  /// @default null
  int? pad_token_id;

  /// The id of the *beginning-of-sequence* token.
  /// @type {number}
  /// @default null
  int? bos_token_id;

  /// The id of the *end-of-sequence* token.
  /// Optionally, use a list to set multiple *end-of-sequence* tokens.
  /// @type {number|number[]}
  /// @default null
  dynamic eos_token_id;

  // Generation parameters exclusive to encoder-decoder models
  /// If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.
  /// @type {number}
  /// @default 0
  int encoder_no_repeat_ngram_size;

  /// If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
  /// @type {number}
  /// @default null
  int? decoder_start_token_id;

  // Wild card
  /// Additional generation kwargs will be forwarded to the `generate` function of the model.
  /// Kwargs that are not present in `generate`'s signature will be used in the model forward pass.
  /// @type {Object}
  /// @default {}
  Map<String, dynamic> generation_kwargs;

  GenerationConfig({
    int? max_length,
    this.max_new_tokens,
    int? min_length,
    this.min_new_tokens,
    dynamic early_stopping,
    this.max_time,
    bool? do_sample,
    int? num_beams,
    int? num_beam_groups,
    this.penalty_alpha,
    bool? use_cache,
    double? temperature,
    int? top_k,
    double? top_p,
    double? typical_p,
    double? epsilon_cutoff,
    double? eta_cutoff,
    double? diversity_penalty,
    double? repetition_penalty,
    double? encoder_repetition_penalty,
    double? length_penalty,
    int? no_repeat_ngram_size,
    this.bad_words_ids,
    this.force_words_ids,
    bool? renormalize_logits,
    this.constraints,
    this.forced_bos_token_id,
    this.forced_eos_token_id,
    bool? remove_invalid_values,
    this.exponential_decay_length_penalty,
    this.suppress_tokens,
    this.streamer,
    this.begin_suppress_tokens,
    this.forced_decoder_ids,
    this.guidance_scale,
    int? num_return_sequences,
    bool? output_attentions,
    bool? output_hidden_states,
    bool? output_scores,
    bool? return_dict_in_generate,
    this.pad_token_id,
    this.bos_token_id,
    this.eos_token_id,
    int? encoder_no_repeat_ngram_size,
    this.decoder_start_token_id,
    Map<String, dynamic>? generation_kwargs,
  }) :  max_length = max_length ?? 20,
        min_length = min_length ?? 0,
        early_stopping = early_stopping ?? false,
        do_sample = do_sample ?? false,
        num_beams = num_beams ?? 1,
        num_beam_groups = num_beam_groups ?? 1,
        use_cache = use_cache ?? true,
        temperature = temperature ?? 1.0,
        top_k = top_k ?? 50,
        top_p = top_p ?? 1.0,
        typical_p = typical_p ?? 1.0,
        epsilon_cutoff = epsilon_cutoff ?? 0.0,
        eta_cutoff = eta_cutoff ?? 0.0,
        diversity_penalty = diversity_penalty ?? 0.0,
        repetition_penalty = repetition_penalty ?? 1.0,
        encoder_repetition_penalty = encoder_repetition_penalty ?? 1.0,
        length_penalty = length_penalty ?? 1.0,
        no_repeat_ngram_size = no_repeat_ngram_size ?? 0,
        renormalize_logits = renormalize_logits ?? false,
        remove_invalid_values = remove_invalid_values ?? false,
        num_return_sequences = num_return_sequences ?? 1,
        output_attentions = output_attentions ?? false,
        output_hidden_states = output_hidden_states ?? false,
        output_scores = output_scores ?? false,
        return_dict_in_generate = return_dict_in_generate ?? false,
        encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size ?? 0,
        generation_kwargs = generation_kwargs ?? {};

  factory GenerationConfig.fromJson(Map<String, dynamic> json) => GenerationConfig(
    max_length: json['max_length'],
    max_new_tokens: json['max_new_tokens'],
    min_length: json['min_length'],
    min_new_tokens: json['min_new_tokens'],
    early_stopping: json['early_stopping'],
    max_time: json['max_time'],
    do_sample: json['do_sample'],
    num_beams: json['num_beams'],
    num_beam_groups: json['num_beam_groups'],
    penalty_alpha: json['penalty_alpha'],
    use_cache: json['use_cache'],
    temperature: json['temperature'],
    top_k: json['top_k'],
    top_p: json['top_p'],
    typical_p: json['typical_p'],
    epsilon_cutoff: json['epsilon_cutoff'],
    eta_cutoff: json['eta_cutoff'],
    diversity_penalty: json['diversity_penalty'],
    repetition_penalty: json['repetition_penalty'],
    encoder_repetition_penalty: json['encoder_repetition_penalty'],
    length_penalty: json['length_penalty'],
    no_repeat_ngram_size: json['no_repeat_ngram_size'],
    bad_words_ids: json['bad_words_ids'],
    force_words_ids: json['force_words_ids'],
    renormalize_logits: json['renormalize_logits'],
    constraints: json['constraints'],
    forced_bos_token_id: json['forced_bos_token_id'],
    forced_eos_token_id: json['forced_eos_token_id'],
    remove_invalid_values: json['remove_invalid_values'],
    exponential_decay_length_penalty: json['exponential_decay_length_penalty'],
    suppress_tokens: json['suppress_tokens'],
    streamer: json['streamer'],
    begin_suppress_tokens: json['begin_suppress_tokens'],
    forced_decoder_ids: json['forced_decoder_ids'],
    guidance_scale: json['guidance_scale'],
    num_return_sequences: json['num_return_sequences'],
    output_attentions: json['output_attentions'],
    output_hidden_states: json['output_hidden_states'],
    output_scores: json['output_scores'],
    return_dict_in_generate: json['return_dict_in_generate'],
    pad_token_id: json['pad_token_id'],
    bos_token_id: json['bos_token_id'],
    eos_token_id: json['eos_token_id'],
    encoder_no_repeat_ngram_size: json['encoder_no_repeat_ngram_size'],
    decoder_start_token_id: json['decoder_start_token_id'],
    generation_kwargs: json['generation_kwargs'],
  );

  Map<String, dynamic> toJson() => {
    'max_length': max_length,
    'max_new_tokens': max_new_tokens,
    'min_length': min_length,
    'min_new_tokens': min_new_tokens,
    'early_stopping': early_stopping,
    'max_time': max_time,
    'do_sample': do_sample,
    'num_beams': num_beams,
    'num_beam_groups': num_beam_groups,
    'penalty_alpha': penalty_alpha,
    'use_cache': use_cache,
    'temperature': temperature,
    'top_k': top_k,
    'top_p': top_p,
    'typical_p': typical_p,
    'epsilon_cutoff': epsilon_cutoff,
    'eta_cutoff': eta_cutoff,
    'diversity_penalty': diversity_penalty,
    'repetition_penalty': repetition_penalty,
    'encoder_repetition_penalty': encoder_repetition_penalty,
    'length_penalty': length_penalty,
    'no_repeat_ngram_size': no_repeat_ngram_size,
    'bad_words_ids': bad_words_ids,
    'force_words_ids': force_words_ids,
    'renormalize_logits': renormalize_logits,
    'constraints': constraints,
    'forced_bos_token_id': forced_bos_token_id,
    'forced_eos_token_id': forced_eos_token_id,
    'remove_invalid_values': remove_invalid_values,
    'exponential_decay_length_penalty': exponential_decay_length_penalty,
    'suppress_tokens': suppress_tokens,
    'streamer': streamer,
    'begin_suppress_tokens': begin_suppress_tokens,
    'forced_decoder_ids': forced_decoder_ids,
    'guidance_scale': guidance_scale,
    'num_return_sequences': num_return_sequences,
    'output_attentions': output_attentions,
    'output_hidden_states': output_hidden_states,
    'output_scores': output_scores,
    'return_dict_in_generate': return_dict_in_generate,
    'pad_token_id': pad_token_id,
    'bos_token_id': bos_token_id,
    'eos_token_id': eos_token_id,
    'encoder_no_repeat_ngram_size': encoder_no_repeat_ngram_size,
    'decoder_start_token_id': decoder_start_token_id,
    'generation_kwargs': generation_kwargs,
  };

  @override
  String toString() => jsonEncode(toJson());
}
