import os
from argparse import ArgumentParser
from mobileo.constants import IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.mm_utils import tokenizer_image_token
from mobileo.conversation import conv_templates

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/mobileo_unified_1.5B")
parser.add_argument("--prompt", type=str, default="a photo of a cute cat")
args = parser.parse_args()

tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to("cuda:0")
image_processor = model.get_vision_tower().image_processor


def infer(prompt):
    qs = "Please generate image based on the following caption: " + prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
    output_image = model.generate_image(input_ids, pixel_values=None)
    return output_image[0]


def main():
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    image_sana = infer(args.prompt)
    save_path = os.path.join(output_dir, "mobileo_gen.png")
    image_sana.save(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
