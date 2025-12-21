from models.encoders.pointnet_encoder import PointNetEncoder
from models.encoders.pointnetlite_encoder import PointNetLiteEncoder
from models.encoders.pointmae_encoder import PointMAEEncoder
from models.encoders.pointnet2_encoder import PointNet2Encoder
from models.encoders.dgcnn_encoder import DGCNNEncoder
from models.encoders.pointpn_encoder import PointPNEncoder

ENCODER_REGISTRY = {
    "pointnet_encoder": PointNetEncoder,
    "pointnetlite_encoder": PointNetLiteEncoder,
    "pointmae_encoder": PointMAEEncoder,
    "pointnet2_encoder": PointNet2Encoder,
    "dgcnn_encoder": DGCNNEncoder,
    "pointpn_encoder": PointPNEncoder
}

def get_encoder(config):
    encoder_name = config["encoder"]["name"]
    encoder_args = config["encoder"].get("args", {})
    
    encoder_class = ENCODER_REGISTRY.get(encoder_name)
    if encoder_class is None:
        raise ValueError(f"Unknown encoder: {encoder_name}")
        
    encoder = encoder_class(**encoder_args)
    return encoder
