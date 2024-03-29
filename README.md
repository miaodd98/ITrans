# ITrans
ITrans: Generative Image Inpainting with Transformers, ChinaMM 2023, Multimedia Systems. https://link.springer.com/article/10.1007/s00530-023-01211-w  
Paper Link: https://link.springer.com/content/pdf/10.1007/s00530-023-01211-w.pdf

# Abstract
Despite significant improvements, convolutional neural network (CNN) based methods are struggling with handling long-range global image dependencies due to their limited receptive fields, leading to an unsatisfactory inpainting performance under complicated scenarios. To address this issue, we propose the Inpainting Transformer (ITrans) network, which combines the power of both self-attention and convolution operations. The ITrans network augments convolutional encoder–decoder structure with two novel designs, i.e. , the global and local transformers. The global transformer aggregates high-level image context from the encoder in a global perspective, and propagates the encoded global representation to the decoder in a multi-scale manner. Meanwhile, the local transformer is intended to extract low-level image details inside the local neighborhood at a reduced computational overhead. By incorporating the above two transformers, ITrans is capable of both global relationship modeling and local details encoding, which is essential for hallucinating perceptually realistic images. Extensive experiments demonstrate that the proposed ITrans network outperforms favorably against state-of-the-art inpainting methods both quantitatively and qualitatively.

# Citation
Miao, W., Wang, L., Lu, H. et al. ITrans: generative image inpainting with transformers. Multimedia Systems 30, 21 (2024). https://doi.org/10.1007/s00530-023-01211-w
