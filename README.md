# caffe-dev-tools
### Usage
First, change `caffe_root` to your caffe path in python file which you want to use.  
##### For BatchNorm absorbing:  
```bash
python bn_absorber.py --model ${DEPLOY_MODEL_PATH}.prototxt \
                      --output ${OUTPUT_PATH_AND_NAME_PREFIX} \
                      --weights ${DEPLOY_WEIGHTS_PATH}.caffemodel \
                      --absorb_weights
```
Then it will produce `${OUTPUT_PATH_AND_NAME_PREFIX}_merge_bn.prototxt` and `${OUTPUT_PATH_AND_NAME_PREFIX}_merge_bn.caffemodel`.

##### For computation complexity and memory access analysis: 
```bash
python compute_complexity.py ${DEPLOY_MODEL_PATH}.prototxt
```
Then the imformation of model parameters, computation complexity and memory access of each layer will display like this:
```
name                	params    	memory    	flops

conv1               	144.00    	13.47K    	112.90K
conv2               	2.30K     	27.39K    	1.81M
conv3               	4.61K     	14.02K    	903.17K
conv4               	9.22K     	21.76K    	1.81M
conv5               	16.13K    	20.44K    	790.27K
conv6               	28.22K    	33.71K    	1.38M
ip2                 	560.00    	626.00    	560.00

 ########### result ###########
#params=61.18K, #FLOPs=6.80M, #Memory Access=979.05K
```
Because of different computing method, this result may not exactly equal to the result reported on other place.
##### For model weights analysis: 
```bash
python model_analysis.py --model ${DEPLOY_MODEL_PATH}.prototxt \
                         --weights ${DEPLOY_WEIGHTS_PATH}.caffemodel \
                         [--display]  # display curve
```
This tool computes the weights' `mean` and `std` for each layer. It futher computes `L1` of each kernel, help you to check how many kernels failed after optimization.  

### tools under folder
You should slightly modify the python files if you want to use `Semantic_Segmentation` tools.  

__ERROR__: Tools in `under_developing` folder are mostly under developing and may failed to work, or only work in some special case. So not recommend to use these.