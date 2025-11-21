import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v3_small
import os
from collections import OrderedDict
import numpy as np
import warnings

# Tắt cảnh báo về số lượng workers
warnings.filterwarnings("ignore", "The given NumPy array is not writeable, and PyTorch does not make a copy.")

# I. THIẾT LẬP VÀ TẢI DATASET PACS

# Thiết lập device (Mặc định dùng 'cuda' như trong file PDF)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Means and standard deviations ImageNet vì network được pretrained
means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Define transforms to apply to each image
transf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), #
    transforms.ToTensor(), #
    transforms.Normalize(means, stds), #
])

# Clone github repository với data
if not os.path.isdir("./Homework3-PACS"):
    print("Cloning PACS dataset...")
    # Lệnh git clone
    os.system("git clone https://github.com/MachineLearning2020/Homework3-PACS")
    print("Done cloning.")

# Define datasets root
DIR_ART = "Homework3-PACS/PACS/art_painting"
DIR_PHOTO = "Homework3-PACS/PACS/photo"
DIR_CARTOON = "Homework3-PACS/PACS/cartoon"
DIR_SKETCH = "Homework3-PACS/PACS/sketch"

# Prepare Pytorch train/test Datasets
photo_dataset = datasets.ImageFolder(DIR_PHOTO, transform=transf)
art_dataset = datasets.ImageFolder(DIR_ART, transform=transf)
cartoon_dataset = datasets.ImageFolder(DIR_CARTOON, transform=transf)
sketch_dataset = datasets.ImageFolder(DIR_SKETCH, transform=transf)

# photo domain được dùng làm tập train, val của photo và toàn bộ domain còn lại dùng để test
train_size = int(0.8 * len(photo_dataset)) #
test_size = len(photo_dataset) - train_size #
train_dataset, test_dataset_photo_val = random_split(photo_dataset, [train_size, test_size]) #

# Concatenate tất cả các dataset test
test_datasets = ConcatDataset([test_dataset_photo_val, art_dataset, cartoon_dataset, sketch_dataset]) #

# Create Dataloaders
batch_size = 128
trainloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
)
testloader = DataLoader(
    test_datasets, batch_size=batch_size, shuffle=False, num_workers=4
)

num_classes = len(photo_dataset.classes)
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_datasets)}")
print(f"Number of classes: {num_classes}")

# 
# II. ĐỊNH NGHĨA CÁC LỚP GNN
# 

### EdgeNet: Dự đoán mối quan hệ/edge giữa 2 sample bất kỳ
class EdgeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio=(1,)):
        super(EdgeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio] #
        self.device = device

        # Define layers
        layer_list = OrderedDict() #
        for l in range(len(num_features_list)):
            in_c = num_features_list[l-1] if l > 0 else in_features #
            out_c = num_features_list[l] #
            
            layer_list[f"conv{l}"] = nn.Conv2d(
                in_channels=in_c, out_channels=out_c, kernel_size=1, bias=False #
            )
            layer_list[f"norm{l}"] = nn.BatchNorm2d(num_features=out_c) #
            layer_list[f"relu{l}"] = nn.LeakyReLU() #

        # Add final similarity kernel
        layer_list["conv_out"] = nn.Conv2d(
            in_channels=num_features_list[-1], out_channels=1, kernel_size=1 #
        )

        self.sim_network = nn.Sequential(layer_list).to(device) #

    def forward(self, node_feat):
        # node_feat: (bs, dim)
        num_tasks = 1 #
        num_data = node_feat.size(0) #
        
        # Thêm dim cho task
        node_feat = node_feat.unsqueeze(dim=0) # (1, bs, dim)

        # Compute difference matrix |x_i - x_j|
        x_i = node_feat.unsqueeze(2) # (1, bs, 1, dim)
        x_j = torch.transpose(x_i, 1, 2) # (1, 1, bs, dim)
        x_ij = torch.abs(x_i - x_j) # (1, bs, bs, dim)
        x_ij = torch.transpose(x_ij, 1, 3) # (1, dim, bs, bs)

        # Compute similarity/dissimilarity (sim_val là raw correlation matrix)
        sim_val = (
            torch.sigmoid(self.sim_network(x_ij)).squeeze(1).squeeze(0) #
            .to(self.device)
        ) # (bs, bs)

        # Normalize affinity matrix
        # Identity matrix: đảm bảo mối liên hệ của mỗi node với chính nó
        force_edge_feat = (
            torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device).squeeze(0)
        ) # (bs, bs)

        # edge_feat = sim_val + Identity
        edge_feat = sim_val + force_edge_feat #
        edge_feat = edge_feat + 1e-6 # add small value to avoid nan

        # Normalize (chuẩn hóa trên mỗi hàng)
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1) # normalize

        return edge_feat, sim_val # (bs, bs), (bs, bs)


### NodeNet: Tổng hợp thông tin từ các node lân cận
class NodeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio=(1,)):
        super(NodeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio] #
        self.device = device #

        # Define layers
        layer_list = OrderedDict() #
        
        # in_channels là in_features * 2 vì ta cat [node_feat, aggr_feat]
        for l in range(len(num_features_list)):
            in_c = num_features_list[l-1] if l > 0 else in_features * 2 #
            out_c = num_features_list[l] #
            
            layer_list[f"conv{l}"] = nn.Conv2d(
                in_channels=in_c, out_channels=out_c, kernel_size=1, bias=False #
            )
            layer_list[f"norm{l}"] = nn.BatchNorm2d(num_features=out_c) #
            
            # Non-linear (LeakyReLU) cho các layer trừ layer cuối
            if l < (len(num_features_list) - 1):
                layer_list[f"relu{l}"] = nn.LeakyReLU()

        self.network = nn.Sequential(layer_list).to(device) #

    def forward(self, node_feat, edge_feat):
        # node_feat: (bs, dim), edge_feat: (bs, bs)
        num_tasks = 1 #
        num_data = node_feat.size(0) #
        
        # Thêm dim cho task
        node_feat_task = node_feat.unsqueeze(0) # (1, bs, dim)

        # Tạo mask: set diagonal=0 để bỏ qua self-loop khi tổng hợp
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device) # (1, bs, bs)
        
        # Chuẩn hóa edge_feat: set diagonal=0 và chuẩn hóa L1 trên chiều cuối cùng
        edge_feat_norm = F.normalize(edge_feat.unsqueeze(0) * diag_mask, p=1, dim=-1) # (1, bs, bs)

        # Compute attention and aggregate (Message Passing)
        aggr_feat = torch.bmm(edge_feat_norm, node_feat_task) # (1, bs, dim)
        aggr_feat = aggr_feat.squeeze(0) # (bs, dim)

        # Concatenate node_feat (ban đầu) và aggregated_feat
        # (bs, 2*dim)
        combined_feat = torch.cat([node_feat, aggr_feat], -1) #
        
        # Non-linear transform: áp dụng mạng MLP/Conv2D 1x1
        # Input cho network phải có shape (1, 2*dim, bs, 1) để dùng Conv2d 1x1 hiệu quả
        network_input = combined_feat.transpose(0, 1).unsqueeze(0).unsqueeze(-1) # (1, 2*dim, bs, 1)

        node_feat_gnn = self.network(network_input) # (1, dim, bs, 1)

        # Xử lý output: (1, dim, bs, 1) -> (bs, dim)
        node_feat_gnn = node_feat_gnn.squeeze(-1).squeeze(0).transpose(0, 1) #

        return node_feat_gnn # (bs, dim)


### GCN: Tích hợp EdgeNet và NodeNet
class GCN(nn.Module):
    def __init__(self, in_features, edge_features, out_feature, device, ratio=(1,)):
        super(GCN, self).__init__()
        
        self.edge_net = EdgeNet(
            in_features=in_features, num_features=edge_features, device=device, ratio=ratio #
        )
        self.node_net = NodeNet(
            in_features=in_features, num_features=out_feature, device=device, ratio=ratio #
        )

        self.mask_val = -1 # mask value for no-gradient edges

    def label2edge(self, targets):
        """convert node labels to affinity mask for backprop""" #
        # targets: (1, bs)
        num_sample = targets.size()[-1] #
        
        # Tạo ma trận label
        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample) # (1, bs, bs)
        label_j = label_i.transpose(1, 2) # (1, bs, bs)
        
        # Ground truth edge: 1 nếu cùng class, 0 nếu khác class
        edge = torch.eq(label_i, label_j).float() #
        
        # Tạo mask cho các label không hợp lệ (giá trị -1)
        target_edge_mask = (
            torch.eq(label_i, self.mask_val) + torch.eq(label_j, self.mask_val)
        ).type(torch.bool) #
        
        # Source edge mask: mask cho các edge hợp lệ (dùng để masked_select)
        source_edge_mask = ~target_edge_mask
        
        # Trích xuất các giá trị edge GT hợp lệ (0 hoặc 1)
        init_edge = edge[source_edge_mask].float()
        
        return init_edge, source_edge_mask.squeeze(0) # edge_gt, edge_mask

    def forward(self, init_node_feat):
        # init_node_feat: feature vector từ backbone (bs, dim)
        
        # 1. Edge prediction: (bs, bs), (bs, bs)
        edge_feat, edge_sim = self.edge_net(init_node_feat) #
        
        # 2. Node aggregation: (bs, num_classes)
        logits_gnn = self.node_net(init_node_feat, edge_feat) #
        
        return logits_gnn, edge_sim # final logits, raw correlation matrix


# 
# III. ĐỊNH NGHĨA MODEL CHÍNH (Tích hợp Backbone và GNN)
# 

class Model(nn.Module):
    def __init__(self, num_classes=7, in_features=576, edge_features=576, device="cuda"):
        super(Model, self).__init__() #
        
        # Backbone: MobileNetV3-Small (pretrained=True)
        self.backbone = mobilenet_v3_small(pretrained=True) #
        
        # Loại bỏ Head/Classifier của MobileNetV3-Small để lấy feature vector
        self.backbone.classifier = nn.Sequential() #

        # GNN/GCN module
        self.gcn = GCN(
            in_features=in_features, # 576
            edge_features=edge_features, # 576
            out_feature=num_classes, # 7
            device=device, # "cuda"
            ratio=(1,), #
        )

    def forward(self, x):
        # Forward qua backbone để lấy feature vector (bs, 576)
        x = self.backbone(x)
        
        # Forward qua GCN
        x, edge_sim = self.gcn(x)
        
        # x: final classification logits, edge_sim: raw correlation matrix
        return x, edge_sim #

# 
# IV. HUẤN LUYỆN MODEL (TRAINING LOOP)
# 

# Khởi tạo model và optimizer
model = Model(num_classes=num_classes, device=device, in_features=576).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Giả định learning rate

# Hàm Loss
criterion_cls = nn.CrossEntropyLoss() #
criterion_edge = nn.BCELoss() # Binary Cross Entropy Loss
lambda_edge = 0.3 # Trọng số cho Cls Loss, đặt ngược lại: 0.3 * Cls + 1.0 * Edge

print("\nStarting Training...")
num_epochs = 10 # Giả định 10 epochs

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0): #
        
        # Move inputs and labels to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, edge_sim = model(inputs) # outputs: logits, edge_sim: raw correlation

        # 1. Classification Loss (Cls loss)
        loss_cls = criterion_cls(outputs, labels)

        # 2. Edge Loss
        # Chuyển labels sang dạng ground truth edge
        edge_gt, edge_mask = model.gcn.label2edge(labels.unsqueeze(dim=0))
        
        # Tính toán Edge Loss chỉ trên các vị trí không bị mask (hợp lệ)
        loss_edge = criterion_edge(
            edge_sim.masked_select(edge_mask), # Raw similarity (predict)
            edge_gt # Ground truth edge (label)
        )

        # Total loss: Loss = lambda_edge * Cls_Loss + Edge_Loss
        # Lưu ý: 0.3 * Cls_Loss + Loss_Edge
        loss = lambda_edge * loss_cls + loss_edge

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() #

        if i % 50 == 49:    # In 50 mini-batches một lần
            print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] Loss: {running_loss / 50:.3f}, CLS: {loss_cls.item():.3f}, EDGE: {loss_edge.item():.3f}')
            running_loss = 0.0

print('Finished Training')