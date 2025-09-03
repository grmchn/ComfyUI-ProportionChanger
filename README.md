# ComfyUI-ProportionChanger

[日本語 README](README_ja.md) | English

> **Note**: This README was automatically generated using [Claude Code](https://claude.ai/code) AI-assisted development tools.

This custom node is created by decomposing and porting the WanVideo UniAnimate DWPose Detector node from [kijai's ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper). The key difference is that instead of using image inputs, it now accepts DWPose KeyPoint data as input, enabling manipulation of body types that cannot be estimated by DWPose alone.

Additionally, the Openpose Editor node from [toyxyz/ComfyUI-ultimate-openpose-editor](https://github.com/toyxyz/ComfyUI-ultimate-openpose-editor) has been similarly decomposed and ported, enabling fine-tuning of individual parts.

## Features

### Node Overview
- **ProportionChanger DWPose Detector**: Detects DWPose KeyPoints from images
- **ProportionChanger Reference**: Transforms proportions to reference poses
- **ProportionChanger DWPose Render**: Converts KeyPoints to images
- **ProportionChanger Params**: Adjusts parameters for individual KeyPoint parts
- **ProportionChanger Interpolator**: Interpolates KeyPoint videos with in-betweening
- **pose_keypoint input**: Converts JSON text to KeyPoints
- **pose_keypoint preview**: Converts KeyPoints to JSON

## Installation
### Install via ComfyUI Manager
1. Search for "ComfyUI-ProportionChanger" in ComfyUI Manager's Custom Nodes Manager and install

2. Restart ComfyUI

### Manual Installation

1. Clone this repository into your `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/grmchn/ComfyUI-ProportionChanger.git
```

2. Install dependencies:
```bash
cd ComfyUI-ProportionChanger
pip install -r requirements.txt
```

3. Restart ComfyUI

## Usage

### Basic Workflow

Please refer to example_workflows.

![proportion_changer_basic2](docs/images/example_basic2.png)

## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Models should be automatically downloaded from HuggingFace. Please ensure DWPose models are in the correct directory.
2. **Incorrect body proportions after transformation with reference image**: Make sure the reference image size matches the pose image size. Fine-tune individual body parts using the "ProportionChanger Params" node.
3. **Nothing displays after transformation with reference image**: The reference image's pose estimation by DWPose has failed. Use OpenposeEditor or similar tools to input parameter values manually.

## Attribution and Credits
### Special Thanks
- **[kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)**
- **[toyxyz/ComfyUI-ultimate-openpose-editor](https://github.com/toyxyz/ComfyUI-ultimate-openpose-editor)**

### License

This project is licensed under **GPL 3.0** due to the combination of source materials from different licenses:

- **Primary source**: [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Apache 2.0)
- **Secondary source**: [toyxyz/ComfyUI-ultimate-openpose-editor](https://github.com/toyxyz/ComfyUI-ultimate-openpose-editor) (GPL 3.0)

When combining code from Apache 2.0 and GPL 3.0 licensed projects, the resulting derivative work must be distributed under GPL 3.0 according to license compatibility rules.

### Copyright Notice

- Original WanVideo UniAnimate DWPose Detector: Copyright by kijai
- ProportionChanger Params functionality: Copyright by toyxyz  
- Modifications and integration: This project's contributors

See the [LICENSE](LICENSE) file for the full GPL 3.0 license terms.