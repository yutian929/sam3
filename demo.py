import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("./assets/images/truck.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="a truck on the road")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# --- Visualization Start ---
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 10))
plt.imshow(image)

# Convert tensors to numpy
masks_np = masks.detach().cpu().numpy()
boxes_np = boxes.detach().cpu().numpy()
scores_np = scores.detach().cpu().numpy()

for i in range(len(boxes_np)):
    # Mask: [1, H, W] -> [H, W]
    # Assuming masks are logits or binary, threshold at 0
    mask = masks_np[i, 0] > 0 
    box = boxes_np[i]
    score = scores_np[i]

    # Draw Mask
    color = np.array([30/255, 144/255, 255/255, 0.6]) # Blue with alpha
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # Mask out background (where mask is False/0)
    mask_image[~mask] = 0
    plt.imshow(mask_image)

    # Draw Box
    x0, y0, x1, y1 = box
    plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='green', facecolor='none', lw=2))
    plt.text(x0, y0, f"{score:.2f}", color='white', fontsize=10, bbox=dict(facecolor='green', alpha=0.5))

plt.axis('off')
plt.savefig("vis_sam3_result.png")
print("Saved visualization to vis_sam3_result.png")
# --- Visualization End ---

# #################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]