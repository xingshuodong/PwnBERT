from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
code_snippet = "int main() { char buffer[10]; strcpy(buffer, input); }"
tokens = tokenizer.tokenize(code_snippet)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
