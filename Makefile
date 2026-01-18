.PHONY: tokens
tokens:
	cargo run --bin tokens

.PHONY: logits
logits:
	cargo run --bin logits --features metal,accelerate

.PHONY: haiku
haiku:
	cargo run --bin haiku --features metal,accelerate

.PHONY: llm-inference
llm-inference:
	cargo run --bin llm-inference --features metal,accelerate
