# 引用Paddlelite预测库
from paddlelite.lite import *

# 1. 创建opt实例
opt=Opt()
# 2. 指定输入模型地址 
# opt.set_model_dir("./hub_model/resnet50_vd_wildanimals_model")
opt.set_model_file("./trained_model/ResNet50_vd_ssld_90/__model__")
opt.set_param_file("./trained_model/ResNet50_vd_ssld_90/__params__")
# opt.set_model_file("./trained_model/MobileNetV3_large_x1_0/__model__")
# opt.set_param_file("./trained_model/MobileNetV3_large_x1_0/__params__")

# 3. 指定转化类型： arm、x86、opencl、xpu、npu
opt.set_valid_places("x86")
# 4. 指定模型转化类型： naive_buffer、protobuf
opt.set_model_type("naive_buffer")
# 4. 输出模型地址
# opt.set_optimize_out("./hub_model/resnet50_vd_wildanimals_x86_model") #如果文件夹不存在就会报错无法生成模型
opt.set_optimize_out("./trained_model/ResNet50_90_trash_x86_model")
# 5. 执行模型优化
opt.run()