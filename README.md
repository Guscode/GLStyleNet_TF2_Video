# GLStyleNet_TF2_Video

This repo includes an updated version of [GLStyleNet](https://github.com/EndyWon/GLStyleNet) (Wang et al., 2018). <b\>

Paper: [GLStyleNet: Higher Quality Style Transfer Combining Global and Local Pyramid Features](https://arxiv.org/abs/1811.07260)

GLStyleNet performs state of the art style-transfer by including feature information in a novel feature pyramid fusion neural network. This version has been updated to be executable with Tensorflow 2, and features for automatic masking and stylizing videos has been added.

## How to use

To run GLStyleNet, create the virtual environment stylevenv:

__Setting up virtual environment and downloading data__
```bash
cd directory/where/you/want/GLStyleNet_TF2_Video
git clone https://github.com/Guscode/GLStyleNet_TF2_Video.git
cd GLStyleNet_TF2_Video
bash create_stylevenv.sh
source stylevenv/bin/activate
```

Running style transfer on test images:
```bash
python GLStyleNet.py --content images/mvdp_win.jpeg  --style images/style.png 
```
### Additional parameters
 

```bash
--content #specify path to content image or video
--content-mask #path to content-mask. if None, content mask is created using masking_functions.py
--content-weight #Weight of content
--style #Path to style image
--style-mask #Path to style mask,if None, content mask is created using masking_functions.py
--local-weight #Weight of local style loss.
--semantic-weight #Weight of semantic map channel.
--global-weight #Weight of global style loss.
--output #Path to output
--iterations #number of iterations, default = 100.
--smoothness #Weight of image smoothing scheme.
--input-type #Specify input type, default = image, options ["image", "video"]
--fps #specify desired frames per second in stylized video, default = 12
--init #Image path to initialize, "noise" or "content" or "style".
--device #Specify devices: "gpu"(default: all gpu) or "gpui"(e.g. gpu0) or "cpu" 
--class-num #number of semantic classes
--start-at #start at specific frame in video, default = 0.
```

# Examples

__Input:__ 
input was the content and style pictures, masks were made automatically with masking_functions.py <b\>
Iterations: 100 <b\>
Duration: 32.4 minutes

<a href="https://github.com/GLStyleNet_TF2_Video">
    <img src="/readme_images/masked_content.png" alt="Logo" width="540" height="202">
</a>

<a href="https://github.com/GLStyleNet_TF2_Video">
    <img src="/readme_images/masked_style.png" alt="Logo" width="540" height="202">
</a>


__Result__

<a href="https://github.com/GLStyleNet_TF2_Video">
    <img src="/readme_images/jonas_stylized.jpg" alt="Logo" width="620" height="410">
</a>
