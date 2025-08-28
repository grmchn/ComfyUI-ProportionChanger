# ComfyUI-ProportionChanger

[日本語 README](README_ja.md) | English

> **Note**: This README was automatically generated using [Claude Code](https://claude.ai/code) AI-assisted development tools.

**Advanced DWPose-based body proportion manipulation and keypoint processing for ComfyUI**

This custom node is created by decomposing and copying the WanVideo UniAnimate DWPose Detector node from [kijai's ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper). The key difference is that instead of using image inputs, it now accepts DWPose KeyPoint data as input, enabling more flexible and detailed body proportion manipulation.

Transform body proportions, manipulate pose keypoints, and create more natural and consistent pose sequences through advanced proportion scaling algorithms.

## Features

### Core Nodes
- **ProportionChanger DWPose Detector**: Basic DWPose detection with keypoint output
- **ProportionChanger Reference**: Apply proportion changes from reference poses
- **ProportionChanger DWPose Render**: Pose visualization and rendering
- **ProportionChanger Params**: Interactive parameter control for pose adjustments
- **ProportionChanger Interpolator**: Frame interpolation for fluid motion
- **pose_keypoint input**: Utility for JSON to POSE_KEYPOINT conversion
- **pose_keypoint preview**: Utility for POSE_KEYPOINT to JSON conversion

### Key Capabilities
- **Proportion Manipulation**: Scale body parts (head, torso, limbs) independently
- **Reference-based Scaling**: Apply proportions from reference poses to target sequences
- **Format Conversion**: Seamless conversion between DWPose and POSE_KEYPOINT formats
- **Batch Processing**: Handle multiple frames and pose sequences efficiently
- **Interactive UI**: Real-time parameter adjustment with pose preview

## Installation

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

## Dependencies

### Required Models
**Automatic Download Feature Available** - Models are automatically downloaded on first run!

**DWPose Models** (automatically placed in `unianimate/models/DWPose/`):
- `dw-ll_ucoco_384_bs5.torchscript.pt` - Main pose detection model
- `yolox_l.torchscript.pt` - Object detection model

**Download Sources**:
- `yolox_l.torchscript.pt`: [hr16/yolox-onnx](https://huggingface.co/hr16/yolox-onnx)
- `dw-ll_ucoco_384_bs5.torchscript.pt`: [hr16/DWPose-TorchScript-BatchSize5](https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5)

> 💡 **No Manual Download Required**: When model files don't exist, they are automatically downloaded from Hugging Face on first node execution.

### Python Dependencies
See `requirements.txt` for complete list. Main dependencies include:
- `torch`
- `torchvision`
- `opencv-python`
- `numpy`

## Usage

### Basic Workflow

1. **Load Image**: Input your source image
2. **DWPose Detection**: Use `ProportionChanger DWPose Detector` to extract pose keypoints
3. **Apply Proportions**: Use `ProportionChanger Reference` with reference pose for transformations
4. **Render Result**: Use `ProportionChanger DWPose Render` to visualize the final pose

### Advanced Features

#### Proportion Reference System
- Load a reference pose with desired proportions
- Apply those proportions to target images automatically
- Supports different canvas sizes with automatic scaling

#### Interactive Parameter Control
- `ProportionChanger Params` provides real-time adjustment
- Preview changes before applying to full sequences
- Fine-tune individual body part scaling

## Node Reference

### ProportionChanger DWPose Detector
**Purpose**: Basic pose detection with keypoint output
**Inputs**: `image`, `score_threshold`
**Outputs**: `pose_keypoint`

### ProportionChanger Reference
**Purpose**: Apply reference proportions to target poses
**Inputs**: `pose_keypoint`, `reference_pose_keypoint`
**Outputs**: `changed_pose_keypoint`

### ProportionChanger Params
**Purpose**: Interactive parameter adjustment for pose manipulation
**Inputs**: `pose_keypoint`, various scaling parameters
**Outputs**: `adjusted_pose_keypoint`

## Technical Implementation

### Core Algorithms
- **DWPose Detection**: Modified from WanVideo UniAnimate implementation
- **Proportion Scaling**: Mathematical transformation of keypoint coordinates
- **Canvas Scaling**: Automatic adjustment for different image dimensions

### Data Formats
- **Input**: Standard image formats (PNG, JPG, etc.)
- **Internal**: POSE_KEYPOINT format with normalized coordinates (0-1 range)
- **Metadata**: Canvas dimensions for accurate coordinate transformation

## Attribution and Credits

This project builds upon excellent work from the open-source community:

### Primary Sources
- **[kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)**: Primary foundation for DWPose detection implementation
- **[toyxyz/ComfyUI-ultimate-openpose-editor](https://github.com/toyxyz/ComfyUI-ultimate-openpose-editor)**: Source for ProportionChanger Params node functionality

### Development Methodology
- **Claude Code**: All development performed using AI-assisted programming
- **Vibe Coding**: Iterative development with AI guidance
- **Issue-driven Development**: Systematic approach to feature implementation and bug fixes

### Special Thanks
- **kijai**: For the robust WanVideoWrapper foundation and DWPose implementation
- **toyxyz**: For the innovative openpose editing framework
- **Anthropic**: For Claude Code development tools
- **Open Source Community**: For continuous inspiration and collaboration

## License

This project is licensed under **GPL 3.0** due to the combination of source materials from different licenses:

- **Primary source**: [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Apache 2.0)
- **Secondary source**: [toyxyz/ComfyUI-ultimate-openpose-editor](https://github.com/toyxyz/ComfyUI-ultimate-openpose-editor) (GPL 3.0)

When combining code from Apache 2.0 and GPL 3.0 licensed projects, the resulting derivative work must be distributed under GPL 3.0 according to license compatibility rules.

**Copyright Notice:**
- Original WanVideo UniAnimate DWPose Detector: Copyright by kijai
- ProportionChanger Params functionality: Copyright by toyxyz  
- Modifications and integration: This project's contributors

See the [LICENSE](LICENSE) file for the full GPL 3.0 license terms.

## Development Status

This project is actively developed with Claude AI assistance. Issues and feature requests are managed through AI-driven development cycles.

### Recent Updates
- Reference-based proportion scaling system
- Batch processing capabilities
- Interactive parameter control
- Canvas size compatibility improvements

## Contributing

While this project is primarily developed using Claude Code methodology, community feedback and issue reports are welcome. Please note that development follows an AI-assisted workflow.

## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Ensure DWPose models are in correct directories
2. **Canvas Size Mismatches**: Check that canvas_width/canvas_height metadata is present
3. **Coordinate Range Issues**: Verify pose data uses normalized coordinates (0-1)

### Debug Information
Enable debug logging in `utils.py` for detailed processing information.

## Example Workflows

Check the `example_workflows/` directory for sample ComfyUI workflows demonstrating various features and use cases.

---

**Note**: This is a specialized tool for pose manipulation and proportion adjustment. For general video generation, consider using the original WanVideoWrapper or native ComfyUI implementations.