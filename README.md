# GPT-Neo za Generiranje Fiktivnih Priča / GPT-Neo_fiction_story_generator

Ovaj model je fino podešena verzija [EleutherAI-jevog GPT-Neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125m) modela, optimiziran za generiranje fiktivnih priča.

Obučen je na skupu podataka dostupnom [ovdje](https://github.com/facebookresearch/fairseq/tree/main/examples/stories).
Link na Paperspace notebook nalazi se [ovdje](https://console.paperspace.com/tincando/notebook/rwb1wo2tukeo4km)

## Upotrebe i ograničenja

Model je dizajniran za generiranje kreativnih fiktivnih priča. Može se koristiti u razne svrhe, uključujući, ali ne ograničavajući se na:

- Pripovijedanje: Generiranje zanimljivih i maštovitih fiktivnih priča.
- Generiranje Sadržaja: Stvaranje sadržaja za blogove, web stranice ili druge medije s elementom pripovijedanja.
- Kreativno Pisanje: Pomoć autorima i piscima pri razmišljanju o idejama i razvijanju narativa.

## Performanse Modela

- Podaci za Obuku: Model je obučen na raznolikom skupu podataka fiktivnih priča i prompteva.
- Metrike Evaluacije: Performanse metrika, kao što su perpleksnost ili BLEU skorovi, mogu varirati ovisno o konkretnom zadatku i skupu podataka.


## Upotreba
Da biste koristili model za generiranje priča u Jupyter bilježnici (Jupyter Notebook), slijedite ove korake:

```
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(Tincando/fiction_story_generator)
model = GPTNeoForCausalLM.from_pretrained(Tincando/fiction_story_generator)
# Generate a fiction story
input_prompt = "[WP] I can't believe I died the same way twice."
input_ids = tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
output = model.generate(input_ids,
        max_length=300,
        temperature=0.9,
        top_k=2,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=2
)
generated_story = tokenizer.batch_decode(output,clean_up_tokenization_spaces=True)[0]
print(generated_story)
```


## About

* Ovaj izvorni kod rezultat je završnog rada na FIPU:
  * Sveučilište Jurja Dobrile u Puli
  * Fakultet informatike u Puli
* Mentor: izv.prof.dr.sc. Darko Etinger
* Student: Tin Kanjovsky
* Tema / naslov: GPT-Neo za Generiranje Fiktivnih Priča

## Značajke

* Generiranje fiktivnih priča
* GPT-Neo model (transformer-based neural networks)
* Natural language processing

## Code

* Python
* Jupyter Notebook
* Datasets: [Fairseq Repository](https://github.com/facebookresearch/fairseq/tree/main/examples/stories)

## Citiranje
Ako koristite ovaj model u svojem radu, molimo razmislite o citiranju originalnog GPT-Neo modela i skupa podataka koji su korišteni za fino podešavanje:

- [GPT-Neo Paper](https://github.com/EleutherAI/gpt-neo)
- [Fairseq Repository](https://github.com/facebookresearch/fairseq/tree/main/examples/stories)
- [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)

## Other

* FIPU: https://fipu.unipu.hr/
* UNIPU: https://www.unipu.hr/

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step   | Validation Loss |
|:-------------:|:-----:|:------:|:---------------:|
| 3.0842        | 1.0   | 34075  | 3.1408          |
| 3.0026        | 2.0   | 68150  | 3.1275          |
| 2.9344        | 3.0   | 102225 | 3.1270          |
| 2.8932        | 4.0   | 136300 | 3.1306          |
| 2.8517        | 5.0   | 170375 | 3.1357          |


### Framework versions

- Transformers 4.28.0
- Pytorch 1.12.1+cu116
- Datasets 2.4.0
- Tokenizers 0.12.1