# Image Debanding using Inversion by Direct Iteration 
This repository implements training and tools to deband images and videos using generative models. 


https://github.com/ksasso1028/indi-debanding/assets/11267645/4def7b00-4bc7-4ca0-9c13-faae985c3c63



# Examples
![banded-example-8](https://github.com/ksasso1028/indi-debanding/assets/11267645/b0e23add-b42f-4680-b9d3-c5f530525c9d)
![restored-example-8](https://github.com/ksasso1028/indi-debanding/assets/11267645/2ac95c37-f75d-4ae7-9be4-0181cab31bf6)

![banded-example-1](https://github.com/ksasso1028/indi-debanding/assets/11267645/8ae4159b-100b-44b0-aa57-72052eef79db)
![restored-example-1](https://github.com/ksasso1028/indi-debanding/assets/11267645/856e951d-ff31-4af2-864c-6168e96b1a57)

![banded-example-6](https://github.com/ksasso1028/indi-debanding/assets/11267645/d1c99c50-19f1-4c15-977b-0caa179e4f8c)
![restored-example-6](https://github.com/ksasso1028/indi-debanding/assets/11267645/1de6ae89-20d4-44c3-8a50-8335570be4ba)

![banded-example-2](https://github.com/ksasso1028/indi-debanding/assets/11267645/5dd6f45f-eef3-4edc-a2b8-ffd53511a3f5)
![restored-example-2](https://github.com/ksasso1028/indi-debanding/assets/11267645/c5da7384-d4a9-47dd-a218-c6b2373e211f)





# Introduction to Image Banding
Image banding is a visual artifact that appears when smooth color gradients are displayed with insufficient color depth, resulting in visible lines or "bands" instead of a seamless transition. This effect is particularly noticeable in areas with gradual color changes, such as skies or shadows. Typically affects cameras with lower bit depth (8 bit footage)

## The Problem
Banding can occur due to various reasons like image compression, low bit-depth formats, or improper processing techniques. It detracts from the visual quality of images and can be problematic in fields requiring high-precision visuals, such as photography, medical imaging, and digital art.

## Solution
This repository offers a solution for image debanding utilizing a technique known as ["Inversion by Direct Iteration"](https://arxiv.org/abs/2303.11435) (INDI) created to handle inverse problems. INDI restores the degraded image in a series of small steps, similar to conditional diffusion models. These models help smooth out gradients and reduce the appearance of bands while maintaining the original details and textures of the image. However instead of applying INDI to the raw signal, we apply it to the 2D FFT of the outputs to enforce restoration across the frequency space. This approach can restore images of variable sizes, with the same model. 

## Importance of Phase and Magnitude in Images
In image processing, phase and magnitude are key aspects of an image's frequency components:

- Magnitude: Indicates the strength of frequency components, affecting image contrast and structure.
- Phase: Contains spatial information, crucial for preserving details and textures.

## Benefits of Optimizing Generative Models in Phase and Magnitude Space
Optimizing generative models in both phase and magnitude spaces improves image quality by:

- Preserving Textures: Phase optimization helps maintain fine details and natural textures.
- Maintaining Structure: Magnitude optimization ensures correct intensity and structure, preventing blurring.
- Enhancing Realism: Combined optimization produces images that are both accurate and visually appealing.
- Heavily penalizes differences in structure and color between the output and target

