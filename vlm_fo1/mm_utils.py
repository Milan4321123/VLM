from PIL import Image
from PIL import ImageDraw, ImageOps
from io import BytesIO
import base64
import re 
import torch
from transformers import StoppingCriteria
from vlm_fo1.constants import IMAGE_TOKEN_INDEX, DEFAULT_REGION_INDEX
import requests
from vlm_fo1.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
    DEFAULT_REGION_TOKEN, 
    DEFAULT_REGION_FEATURE_TOKEN
)
import torch
from transformers import TextStreamer
import random
import re
from typing import List, Tuple
import io
import base64


def _special_token_ids(tokenizer, token):
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Tokenizer does not expose {token} as a single special token.")
        return token_ids
    return [token_id]


def _encode_text(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _uses_qwen35_chat_template(tokenizer):
    template = getattr(tokenizer, "chat_template", "") or ""
    name = getattr(tokenizer, "name_or_path", "") or ""
    return "qwen3.5" in name.lower() or "enable_thinking" in template


def _assistant_generation_suffix(tokenizer, enable_thinking=False):
    # Qwen2.5-VL's template opens the reply with just "<|im_start|>assistant\n".
    # Qwen3.5's inserts a think block right after that marker, and the model
    # expects it. We build prompts by hand (the image/region sentinels rule out
    # apply_chat_template), so reproduce the same suffix here.
    if not _uses_qwen35_chat_template(tokenizer):
        return ""
    if enable_thinking:
        return "<think>\n"
    return "<think>\n\n</think>\n\n"


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """
    Tokenizes prompts containing <image> or <image_0>... special tokens.

    If the prompt uses <image_0>, <image_1>, ..., each is replaced with a placeholder index (-200).
    If the prompt uses <image>, it is replaced with image_token_index.

    Args:
        prompt (str): The prompt potentially containing image tokens.
        tokenizer: The tokenizer object.
        image_token_index (int): Token id to use when encountering <image> token.
        return_tensors (Optional[str]): If 'pt', return a torch tensor.

    Returns:
        List[int] or torch.Tensor: The tokenized input with image token indices inserted appropriately.
    """
    if "<image_0>" in prompt:
        # Case: prompt contains indexed image tokens like <image_0>, <image_1>, etc.
        image_token_pattern = re.compile(r"<image_(\d+)>")
        prompt_chunks = re.split(r'<image_[0-9]+>', prompt)
        image_tags = image_token_pattern.findall(prompt)

        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(tokenizer(chunk).input_ids)
            if i < len(image_tags):
                # Insert placeholder where <image_n> token was.
                input_ids.append(-200)
    else:
        # Case: prompt contains plain <image> tokens.
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            # Helper function to insert a separator token between chunks.
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        # If first chunk starts with <bos> token, make sure to keep it only once.
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        # Insert image_token_index between chunks.
        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])
    # Optionally convert output to PyTorch tensor.
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        else:
            raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids

def tokenizer_image_region_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, region_token_index=DEFAULT_REGION_INDEX, return_tensors=None):
    """
    Tokenizes prompts containing both <image> and <regionfeat> delimiters, inserting specified token indices.

    Each <image> chunk is split, and within that chunk, <regionfeat> locations receive region_token_index.

    Args:
        prompt (str): The prompt with <image> and <regionfeat> delimiters.
        tokenizer: The tokenizer object.
        image_token_index (int): Insert this at <image> splits.
        region_token_index (int): Insert this at <regionfeat> splits.
        return_tensors (Optional[str]): If 'pt', return torch tensor.

    Returns:
        List[int] or torch.Tensor: The tokenized input with region/image tokens placed.
    """
    # Split by <image> tags first.
    image_chunks = prompt.split('<image>')
    
    prompt_chunks = []
    for chunk in image_chunks:
        # Split each image chunk by <regionfeat>.
        obj_chunks = chunk.split('<regionfeat>')
        # Tokenize each subchunk.
        token_chunks = [tokenizer(c).input_ids for c in obj_chunks]
        prompt_chunks.append(token_chunks)
    
    input_ids = []
    offset = 0

    # If first chunk starts with <bos> token, include only once.
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and len(prompt_chunks[0][0]) > 0 and prompt_chunks[0][0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0][0])
    
    # Stitch together all chunks with region/image tokens at appropriate locations.
    for i, chunk_group in enumerate(prompt_chunks):
        if len(chunk_group) > 0:
            input_ids.extend(chunk_group[0][offset:])
        for chunk in chunk_group[1:]:
            input_ids.append(region_token_index)
            input_ids.extend(chunk)
        # Insert <image> token except after the last image chunk.
        if i < len(prompt_chunks) - 1:
            input_ids.append(image_token_index)
    # Optionally convert to PyTorch tensor.
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        else:
            raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids

class KeywordsStoppingCriteria(StoppingCriteria):
    """
    Implements custom stopping criteria for generation based on keywords:
    If the generated output contains any of the keywords, generation stops.
    """
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            # Remove BOS if present except for single token
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        # Track the generation start length
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Checks if a keyword exists in the latest generated output ids for a single batch element.
        """
        offset = min(max(0, output_ids.shape[1] - self.start_len), self.max_keyword_len)
        if offset == 0:
            return False
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Checks for keywords in each batch item; stops when all have satisfied the keyword condition.
        """
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)

def load_image(image_file):
    """
    Loads an image from a local path, base64 string, URL, or PIL.Image.

    Preserve the original dimensions so images and bounding boxes can be
    resized together by resize_shortest_edge_images_and_bboxes.

    Args:
        image_file (str or PIL.Image.Image): Image source.

    Returns:
        PIL.Image.Image: Loaded image in RGB mode at its original size.
    """
    if isinstance(image_file, Image.Image):
        image = image_file.convert("RGB")
    # Case: load from URL
    elif image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    # Case: load from base64-encoded string
    elif image_file.startswith("data:image/"):
        image = image_file.replace("data:image/jpeg;base64,", "")
        image_data = base64.b64decode(image)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    elif isinstance(image_file, str):
        # Case: load from local file path
        image = Image.open(image_file).convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image_file)}")
    
    return image

def image_to_base64(img_pil):
    """
    Encodes a PIL Image as JPEG in base64 format.

    Args:
        img_pil (PIL.Image.Image): Source image.

    Returns:
        str: base64-encoded JPEG image string.
    """
    with io.BytesIO() as buffer:
        img_pil.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def draw_bboxes_and_save(
    image: Image.Image,
    fo1_bboxes: dict = {},
    detection_bboxes: List[Tuple[int, int, int, int]] = [],
    output_path: str = 'output.jpg',
    color: str = 'red',
    total_color: str = 'green',
    width: int = 2
) -> None:
    """
    Draws bounding boxes (both ground-truth/proposed and detection) on a PIL image and saves result.

    Args:
        image (PIL.Image.Image): Input PIL image object.
        fo1_bboxes (dict): Label -> List[bbox] mapping for annotation bboxes.
        detection_bboxes (List[Tuple]): List of detection bounding boxes; each bbox is (x_min, y_min, x_max, y_max).
        output_path (str): Path to save the output image.
        color (str): Color for fo1_bboxes.
        total_color (str): Color for detection_bboxes.
        width (int): Rectangle outline width.

    Returns:
        None
    """
    draw = ImageDraw.Draw(image)
    
    # Draw detection boxes with `total_color`
    for bbox in detection_bboxes:
        if len(bbox) != 4:
            print(f"Warning: skip the invalid bbox {bbox}")
            continue
        shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
        draw.rectangle(shape, outline=total_color, width=width)

    # Draw annotated bboxes with labels and `color`
    for bbox_label, bbox_list in fo1_bboxes.items():
        for bbox in bbox_list:
            if len(bbox) != 4:
                print(f"Warning: skip the invalid bbox {bbox}")
                continue
            shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
            draw.rectangle(shape, outline=color, width=width)
            draw.text((bbox[0], bbox[1]), bbox_label, fill=color)

    # Propagate write failures so callers cannot claim an artifact was saved.
    image.save(output_path)

def adjust_bbox(bbox_list, original_h, original_w, resize_h, resize_w):
    """
    Adjusts bounding boxes from original image size to resized image size, compensating for scaling.

    Args:
        bbox_list (List[List[float]]): List of original boxes [x1, y1, x2, y2].
        original_h (int): Original image height.
        original_w (int): Original image width.
        resize_h (int): Resized image height.
        resize_w (int): Resized image width.

    Returns:
        List[List[float]]: Bounding boxes transformed to resized image coordinates.
    """
    output_list = []
    def adjust_bbox_range(bbox, width, height):
        # Ensure all coordinates are within the original image border.
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        return [x1, y1, x2, y2]

    for bbox in bbox_list:
        bbox = adjust_bbox_range(bbox, original_w, original_h)
        bbox[0] = bbox[0] * resize_w / original_w
        bbox[1] = bbox[1] * resize_h / original_h
        bbox[2] = bbox[2] * resize_w / original_w
        bbox[3] = bbox[3] * resize_h / original_h
        output_list.append(bbox)
    return output_list

def extract_predictions_to_bboxes(prediction: str, bbox_list):
    """
    Parse prediction string in the expected format and map each ground label
    to its corresponding bounding boxes using bbox_list.

    Args:
        prediction (str): Model output string with <ground>...<objects>... markup.
        bbox_list (List[List[float]]): Full list of predicted or reference bounding boxes.

    Returns:
        dict: label -> list of bboxes
    """
    label_to_indexes = {}
    label_to_bboxes = {}

    match_pattern = r"<ground>(.*?)<\/ground><objects>(.*?)<\/objects>"
    matches = re.findall(match_pattern, prediction)

    for label_text, indexes in matches:
        label_text = label_text.strip()
        indexes_tags = re.findall(r"<region\d+>", indexes)
        region_indexes = set([int(index.split("<region")[-1].split(">")[0]) for index in indexes_tags])
        if label_text not in label_to_indexes:
            label_to_indexes[label_text] = region_indexes
        else:
            label_to_indexes[label_text] = label_to_indexes[label_text] | region_indexes

    for label, indexes in label_to_indexes.items():
        label_to_bboxes[label] = [bbox_list[index] for index in indexes]

    return label_to_bboxes

def extract_predictions_to_indexes(prediction: str):
    """
    Parse prediction string, returning label -> set-of-indexes mapping.

    Args:
        prediction (str): Model prediction output.

    Returns:
        dict: label -> set(int)
    """
    label_to_indexes = {}
    match_pattern = r"<ground>(.*?)<\/ground><objects>(.*?)<\/objects>"
    matches = re.findall(match_pattern, prediction)

    for label_text, indexes in matches:
        label_text = label_text.strip()
        indexes_tags = re.findall(r"<region\d+>", indexes)
        region_indexes = set([int(index.split("<region")[-1].split(">")[0]) for index in indexes_tags])
        if label_text not in label_to_indexes:
            label_to_indexes[label_text] = region_indexes
        else:
            label_to_indexes[label_text] = label_to_indexes[label_text] | region_indexes

    return label_to_indexes

def resize_shortest_edge_images_and_bboxes(
    image_list: List[Image.Image],
    bbox_lists: List,
    candidate_sizes=None,
    max_size: int = 2048,
):
    """Resize images and boxes together; a single image may have zero boxes."""
    if not image_list:
        raise ValueError("At least one image is required.")
    flat_input = len(image_list) == 1 and (
        not bbox_lists or (bbox_lists[0] and isinstance(bbox_lists[0][0], (int, float)))
    )
    if bbox_lists is None or len(bbox_lists) == 0:
        normalized_boxes = [[] for _ in image_list]
    elif flat_input:
        normalized_boxes = [bbox_lists]
    else:
        normalized_boxes = bbox_lists
    if len(image_list) != len(normalized_boxes):
        raise ValueError("Each image must have its own bounding box list.")
    target_size = random.choice(candidate_sizes) if candidate_sizes else None
    resized_images, transformed = [], []
    for img, boxes in zip(image_list, normalized_boxes):
        width, height = img.size
        scale = target_size / min(width, height) if target_size else 1.0
        scale = min(scale, max_size / max(width, height))
        new_width, new_height = max(28, int(width * scale)), max(28, int(height * scale))
        resized_images.append(img.resize((new_width, new_height), Image.Resampling.BICUBIC))
        transformed.append([
            [x1 * new_width / width, y1 * new_height / height,
             x2 * new_width / width, y2 * new_height / height]
            for x1, y1, x2, y2 in boxes
        ])
    return resized_images, transformed[0] if flat_input else transformed


def make_message_context(tokenizer, message, chat_format="chatml", vision_tokens=None):
    """Build one chat turn while retaining image and region sentinels."""
    if chat_format != "chatml":
        raise ValueError("Only chatml is supported.")
    role = message.get("role")
    if role not in {"system", "user", "assistant"}:
        raise ValueError("Message role must be system, user or assistant.")
    content = message["content"]
    boxes = message.get("bbox_list")
    image_urls = []
    start, end = vision_tokens or (DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks = []
        for part in content:
            if part["type"] == "text":
                chunks.append(part["text"])
            elif part["type"] == "image_url":
                chunks.append(start + "<image>" + end + "\n")
                image_urls.append(part["image_url"]["url"])
                if boxes:
                    chunks.append("".join(
                        DEFAULT_REGION_TOKEN.replace("<i>", str(i)) + DEFAULT_REGION_FEATURE_TOKEN
                        for i in range(len(boxes))
                    ) + "\n")
            else:
                raise ValueError(f"Unsupported content type: {part['type']}")
        text = "".join(chunks)
    else:
        raise ValueError("Message content must be text or a list of text/image parts.")
    prompt = f"<|im_start|>{role}\n{text}<|im_end|>\n"
    ids = tokenizer_image_region_token(prompt, tokenizer) if boxes else tokenizer_image_token(prompt, tokenizer)
    return prompt, ids, image_urls, boxes

def prepare_inputs(model_name, model, image_processors, tokenizer, messages, device="cuda", max_tokens=512, top_p=1.0, temperature=0.0, do_sample=False, image_size=None, enable_thinking=False, stream=True, verbose=True):
    """
    Fully prepares keyword arguments for model.generate (and compatible API) from messages and model specs.

    Handles prompt assembly, tokenization, image loading/preprocessing, region support, streaming, etc.
    Supports specific tweak for Qwen2.5-VL style vision tokens.

    Args:
        model_name (str): Model identifier string.
        model: Model/config object.
        image_processors (tuple): (primary, auxiliary) image processors.
        tokenizer: Tokenizer object.
        messages (list): Multi-message input list (chat history).
        device (str): Target (usually 'cuda' or 'cpu').
        max_tokens, top_p, temperature, do_sample: Standard generation kwargs.

    Returns:
        dict: ready-to-use argument dict for model.generate().
    """
    if not messages:
        raise ValueError("At least one message is required.")
    vision_tokens = None
    if "qwen" in model_name.lower() or "qwen" in getattr(model.config, "model_type", ""):
        vision_tokens = ("<|vision_start|>", "<|vision_end|>")

    primary_image_processor, auxiliary_image_processor = image_processors

    prompt = ""
    input_tokens = []
    image_urls = []
    bbox_list = None
    # Compose prompt and accumulate all components from provided messages
    for message in messages:
        boxes = message.get("bbox_list")
        if boxes and len(boxes) > getattr(model.config, "mm_num_region_tokens", 100):
            raise ValueError("The FO1 checkpoint supports at most 100 proposal regions per image.")
        inp, context_tokens, message_images, message_boxes = make_message_context(
            tokenizer, message, vision_tokens=vision_tokens
        )
        if message_images:
            image_urls.extend(message_images)
            bbox_list = message_boxes
        prompt += inp
        input_tokens.extend(context_tokens)
    if len(image_urls) > 1:
        raise ValueError("This inference entry point supports one image per request.")

    # Ensure a system prompt at start, if not already present.
    if not any(message.get("role") == "system" for message in messages):
        system_content = "system\nYou are a helpful assistant."
        system_prompt = "<|im_start|>" + system_content + "<|im_end|>" + "\n"
        prompt = system_prompt + prompt
        system_tokens = (
            _special_token_ids(tokenizer, "<|im_start|>")
            + _encode_text(tokenizer, system_content)
            + _special_token_ids(tokenizer, "<|im_end|>")
            + _encode_text(tokenizer, "\n")
        )
        input_tokens = system_tokens + input_tokens

    # Ensure prompt ends with assistant's turn.
    if not prompt.endswith("<|im_start|>assistant"):
        assistant_suffix = _assistant_generation_suffix(tokenizer, enable_thinking=enable_thinking)
        last_assistant_prompt = "<|im_start|>" + "assistant" + "\n" + assistant_suffix
        prompt += last_assistant_prompt
        last_assistant_tokens = _special_token_ids(tokenizer, "<|im_start|>") + _encode_text(tokenizer, "assistant\n" + assistant_suffix)
        input_tokens.extend(last_assistant_tokens)

    primary_images_tensor = None
    auxiliary_images_tensor = None
    primary_image_grid_thws = None
    images = []
    if image_urls:
        # Load images, resize them, and update bbox_list downstream
        images = [load_image(i) for i in image_urls]
        if image_size is not None:
            images, bbox_list = resize_shortest_edge_images_and_bboxes(images, bbox_list, candidate_sizes=[image_size], max_size=2048)
        else:
            images, bbox_list = resize_shortest_edge_images_and_bboxes(images, bbox_list, max_size=2048)

    
        # When region-indexed tokens are enabled
        if getattr(model.config, 'mm_use_region_index_token', False):
            origin_image_size = [image.size for image in images]
            aux_images = images.copy()
            auxiliary_images_tensor = [auxiliary_image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0].to(device) for i in aux_images]

            if bbox_list and len(bbox_list) > 0:
                # Limit number of bbox (for computational constraints, etc.)
                resize_h, resize_w = auxiliary_images_tensor[0].shape[-2:]
                original_w, original_h = origin_image_size[0]
                # Adjust bbox to match resized images (post pre-processing)
                bbox_list = adjust_bbox(bbox_list, original_h, original_w, resize_h, resize_w)
                bbox_list = [torch.tensor(bbox_list, dtype=torch.float32, device=device)]
            else:
                bbox_list = None
        else:
            auxiliary_images_tensor = None

    # Preprocess primary images for main vision model branch
    primary_images = []
    primary_image_grid_thws = []
    for im in images:
        processed_data = primary_image_processor.preprocess(im, return_tensors="pt")
        image_i = processed_data['pixel_values']
        image_grid_thw_i = processed_data['image_grid_thw']
        primary_images.append(image_i)
        primary_image_grid_thws.append(image_grid_thw_i)
    primary_images_tensor = [image_i.to(device) for image_i in primary_images]

    # Stop on the tokenizer's eos; fall back to <|im_end|> for Qwen
    # tokenizers that define no eos token.
    input_ids = torch.tensor([input_tokens]).to(device)
    keywords = []
    if tokenizer.eos_token is not None:
        keywords.append(tokenizer.eos_token)
    elif "qwen" in model_name.lower():
        keywords.append("<|im_end|>")

    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids) if keywords else None
    streamer = TextStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    ) if stream else None

    # Default: greedy decoding if temperature=0. Else: enable sampling.
    if temperature == 0.0:
        do_sample = False
    else:
        do_sample = True

    if verbose:
        print("question:================\n", prompt, "\n=================")
    # print("input ids:========", input_ids, "========")
    generation_kwargs = dict(
        inputs=input_ids,
        images=primary_images_tensor,
        images_aux=auxiliary_images_tensor,
        image_grid_thws=primary_image_grid_thws,
        bbox_list=bbox_list,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_tokens,
        streamer=streamer,
        top_p=top_p,
        use_cache=True,
        stopping_criteria=[stopping_criteria] if stopping_criteria is not None else None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    return generation_kwargs
