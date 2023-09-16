import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import SimpleITK as sitk
from src.base_model import base_model
from src.base_cls_weight_model import base_cls_weight_model
from src.base_seg_weight_model import base_seg_weight_model
from src.our_seg_cls_weight_model import our_seg_cls_weight_model
from train_utils.train_and_eval import evaluate
from my_dataset import SSOCTADataset


def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/ours/epoch_40.pth"

    # 获取设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 加载模型
    model = our_seg_cls_weight_model(class_num=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    # 加载数据
    trans = transforms.Compose([transforms.ToTensor()])
    test_dataset = SSOCTADataset(root='data', train=False, test=True, transforms=trans)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    # 测试集指标评估
    acc, pre, rec, f1_score, confmat, dice = evaluate(model, test_loader, device=device, num_classes=classes+1)
    val_info = str(confmat)
    print(f"acc: {acc}, pre: {pre.mean()}, rec: {rec.mean()}, f1-score: {f1_score}")
    print(val_info)

    # 测试集结果可视化
    images = sitk.ReadImage('./data/test/most_1_type_image_test.nii.gz')
    data = np.array(sitk.GetArrayFromImage(images))
    print('predict size:', data.shape)

    for i in range(len(data)):
        img = trans(data[i, :, :])
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 1, img_height, img_width), device=device)
            model(init_img)

            output = model(img.to(device))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.save(f"./test_result/ours/test_result_{i}.png")

    to_nii(len(data))


# 将测试结果写入到 nii.gz 文件中
def to_nii(size):
    data_root = os.path.join('test_result', 'ours')
    image_all = []
    for i in range(size):
        img_name = os.path.join(data_root, f'test_result_{i}.png')
        img = Image.open(img_name)
        image_all.append(np.array(img))
    image_all = sitk.GetImageFromArray(image_all)
    sitk.WriteImage(image_all, os.path.join(data_root, 'pred_result_ours.nii.gz'))


if __name__ == '__main__':
    main()
