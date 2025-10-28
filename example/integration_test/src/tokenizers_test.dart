import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:jinja_minimal/jinja_minimal.dart';
import 'package:transformers/transformers.dart';

void main() async {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  await Transformers.ensureInitialized(throwOnFail: true);

  test('Able to apply chat template', () async {
    final tokenizer = await AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1");

    const List<Message> chat = [
      Message(role: 'user', content: 'Hello, how are you?'),
      Message(role: 'assistant', content: "I'm doing great. How can I help you today?"),
      Message(role: 'user', content: "I'd like to show off how chat templating works!"),
    ];

    final String text = await tokenizer.apply_chat_template(chat, ApplyChatTemplateOptions(
      tokenize: false,
    ));

    expect(
      text,
      "<s> [INST] Hello, how are you? [/INST] I'm doing great. How can "
          "I help you today?</s> [INST] I'd like to show off how chat "
          "templating works! [/INST]",
    );

    final List<int> input_ids = await tokenizer.apply_chat_template(chat, ApplyChatTemplateOptions(
      tokenize: true,
      return_tensor: false,
    ));

    expect(
      input_ids,
      [
        1, 733, 16289, 28793, 22557, 28725, 910, 460, 368, 28804, 733, 28748,
        16289, 28793, 315, 28742, 28719, 2548, 1598, 28723, 1602, 541, 315,
        1316, 368, 3154, 28804, 2, 733, 16289, 28793, 315, 28742, 28715, 737,
        298, 1347, 805, 910, 10706, 5752, 1077, 3791, 28808, 733, 28748, 16289,
        28793
      ],
    );
  });

  test('Able to apply chat template with nested content', () async {
    final templateString = """{%- for message in messages -%}
{{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' -}}
{%- if message['content'] is string -%}
{{- message['content'] -}}
{%- else -%}
{%- for part in message['content'] -%}
{%- if part['type'] == 'text' -%}
{{- part['text'] -}}
{%- elif part['type'] == 'image' -%}
{{- '<image>' -}}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{- '<|end_of_text|>
' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|start_of_role|>assistant' -}}
{%- if controls -%}{{- ' ' + controls | tojson() -}}{%- endif -%}
{{- '<|end_of_role|>' -}}
{%- endif -%}
""";
    final template = Template(templateString);
    final expected = '<|start_of_role|>user<|end_of_role|><image>Convert this page to docling.<|end_of_text|>\n';

    String text = template.render({
      'messages': [
        {
          'role': 'user',
          'content': [
            { 'type': 'image' },
            { 'type': 'text', 'text': 'Convert this page to docling.' },
          ],
        },
      ],
    });
    print(text);

    expect(text, expected);

    const messages = [
      Message(
        role: 'user',
        content: [
          MessageContent(type: 'image'),
          MessageContent(type: 'text', text: 'Convert this page to docling.'),
        ],
      ),
    ];
    text = template.render({ 'messages': messages });

    expect(text, expected);
  });
}
