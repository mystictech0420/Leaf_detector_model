"""Giao diện web cục bộ hỗ trợ cắt ảnh bằng 2 điểm chọn và nhận dạng bệnh trên lá cây."""

from pathlib import Path
import sys

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

thu_muc_du_an = Path(__file__).resolve().parent
if str(thu_muc_du_an) not in sys.path:
    sys.path.insert(0, str(thu_muc_du_an))

from src.dataset import PROJECT_ROOT
from src.model import build_disease_classifier
from src.predict import classify_prediction


class BoNhanDangBenhLaCay:
    """Lớp quản lý vòng đời mô hình và thực hiện suy luận trên hình ảnh."""

    def __init__(self, duong_dan_mo_hinh: Path) -> None:
        self.duong_dan_mo_hinh = duong_dan_mo_hinh.expanduser().resolve()
        if not self.duong_dan_mo_hinh.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy tệp trọng số mô hình tại: {self.duong_dan_mo_hinh}"
            )

        self.thiet_bi_tinh_toan = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        du_lieu_kiem_tra = torch.load(
            self.duong_dan_mo_hinh,
            map_location="cpu",
            weights_only=True,
        )
        if du_lieu_kiem_tra.get("architecture") != "resnet50":
            raise ValueError(
                f"Kiến trúc mô hình không được hỗ trợ: {self.duong_dan_mo_hinh}"
            )

        self.danh_sach_ten_lop: list[str] = du_lieu_kiem_tra["class_names"]
        so_luong_lop: int = du_lieu_kiem_tra["num_classes"]

        self.mo_hinh = build_disease_classifier(
            num_classes=so_luong_lop,
            pretrained=False,
        ).to(self.thiet_bi_tinh_toan)
        self.mo_hinh.load_state_dict(du_lieu_kiem_tra["state_dict"])
        self.mo_hinh.eval()

        self.bo_tien_xu_ly = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def du_doan_hinh_anh(
        self,
        hinh_anh_dau_vao: Image.Image | None,
    ) -> tuple[str, dict[str, float]]:
        """Nhận ảnh đầu vào và trả về kết quả phân tích cùng bảng xác suất."""
        if hinh_anh_dau_vao is None:
            return "Vui lòng tải lên hoặc dán (Ctrl+V) một hình ảnh chiếc lá.", {}

        hinh_anh_rgb = hinh_anh_dau_vao.convert("RGB")
        tensor_dau_vao = (
            self.bo_tien_xu_ly(hinh_anh_rgb)
            .unsqueeze(0)
            .to(self.thiet_bi_tinh_toan)
        )

        with torch.inference_mode():
            xac_suat_cac_lop = torch.softmax(self.mo_hinh(tensor_dau_vao), dim=1)[0]
            gia_tri_cao_nhat, chi_so_cao_nhat = xac_suat_cac_lop.topk(3)

        danh_sach_du_doan: list[tuple[str, float]] = [
            (self.danh_sach_ten_lop[chi_so], float(xac_suat))
            for xac_suat, chi_so in zip(
                gia_tri_cao_nhat.cpu().tolist(),
                chi_so_cao_nhat.cpu().tolist(),
            )
        ]

        nhan_cao_nhat, xac_suat_cao_nhat = danh_sach_du_doan[0]
        trang_thai, ten_loai_cay, ten_benh = classify_prediction(nhan_cao_nhat)

        if ten_benh is None:
            thong_tin_benh = "Không phát hiện dấu hiệu bệnh (Lá khỏe mạnh)"
        else:
            thong_tin_benh = ten_benh

        noi_dung_bao_cao = (
            f"Thiết bị xử lý: {self.thiet_bi_tinh_toan.type.upper()}\n"
            f"Trạng thái đánh giá: {trang_thai}\n"
            f"Loại cây trồng: {ten_loai_cay}\n"
            f"Tình trạng bệnh lý: {thong_tin_benh}\n"
            f"Độ tin cậy dự đoán: {xac_suat_cao_nhat:.2%}"
        )

        bang_xac_suat_top_3: dict[str, float] = {
            nhan: xac_suat for nhan, xac_suat in danh_sach_du_doan
        }

        return noi_dung_bao_cao, bang_xac_suat_top_3


def khoi_chay_giao_dien() -> None:
    """Khởi tạo bộ nhận dạng và mở giao diện web cục bộ hỗ trợ cắt 2 điểm."""
    duong_dan_checkpoint_mac_dinh = (
        PROJECT_ROOT / "models" / "plant_disease_resnet50.pt"
    )
    bo_nhan_dang = BoNhanDangBenhLaCay(
        duong_dan_mo_hinh=duong_dan_checkpoint_mac_dinh
    )

    def xu_ly_khi_tai_anh_moi(
        hinh_anh_moi: Image.Image | None,
    ) -> tuple[Image.Image | None, None, Image.Image | None, str, dict[str, float]]:
        if hinh_anh_moi is None:
            return None, None, None, "Chưa có hình ảnh nào được tải lên.", {}
        bao_cao, bang_xac_suat = bo_nhan_dang.du_doan_hinh_anh(hinh_anh_moi)
        return hinh_anh_moi, None, hinh_anh_moi, bao_cao, bang_xac_suat

    def xu_ly_click_cat_anh(
        hinh_anh_goc: Image.Image | None,
        diem_goc_thu_nhat: tuple[int, int] | None,
        su_kien_chon: gr.SelectData,
    ) -> tuple[tuple[int, int] | None, Image.Image | None, str, dict[str, float]]:
        if hinh_anh_goc is None:
            return None, None, "Vui lòng tải ảnh lên trước khi chọn vùng cắt.", {}

        toa_do_x_hien_tai = int(su_kien_chon.index[0])
        toa_do_y_hien_tai = int(su_kien_chon.index[1])

        if diem_goc_thu_nhat is None:
            thong_bao_cho = (
                f"Đã ghim góc thứ nhất tại tọa độ ({toa_do_x_hien_tai}, {toa_do_y_hien_tai}).\n"
                "Hãy click chuột vào góc đối diện của chiếc lá để hoàn tất cắt ảnh..."
            )
            return (toa_do_x_hien_tai, toa_do_y_hien_tai), gr.skip(), thong_bao_cho, gr.skip()

        toa_do_x_thu_nhat, toa_do_y_thu_nhat = diem_goc_thu_nhat
        bien_trai = min(toa_do_x_thu_nhat, toa_do_x_hien_tai)
        bien_tren = min(toa_do_y_thu_nhat, toa_do_y_hien_tai)
        bien_phai = max(toa_do_x_thu_nhat, toa_do_x_hien_tai)
        bien_duoi = max(toa_do_y_thu_nhat, toa_do_y_hien_tai)

        if bien_phai - bien_trai < 10 or bien_duoi - bien_tren < 10:
            return None, gr.skip(), "Vùng chọn quá nhỏ, vui lòng click lại 2 góc rộng hơn.", gr.skip()

        hinh_anh_da_cat = hinh_anh_goc.crop((bien_trai, bien_tren, bien_phai, bien_duoi))
        bao_cao, bang_xac_suat = bo_nhan_dang.du_doan_hinh_anh(hinh_anh_da_cat)
        return None, hinh_anh_da_cat, bao_cao, bang_xac_suat

    def xu_ly_hoan_tac_anh_goc(
        hinh_anh_goc: Image.Image | None,
    ) -> tuple[None, Image.Image | None, str, dict[str, float]]:
        if hinh_anh_goc is None:
            return None, None, "Chưa có ảnh gốc để hoàn tác.", {}
        bao_cao, bang_xac_suat = bo_nhan_dang.du_doan_hinh_anh(hinh_anh_goc)
        return None, hinh_anh_goc, bao_cao, bang_xac_suat

    with gr.Blocks(title="Hệ thống Nhận dạng Bệnh trên Lá cây (ResNet-50)") as giao_dien:
        gr.Markdown("# Hệ thống Nhận dạng Bệnh trên Lá cây (ResNet-50)")
        gr.Markdown(
            "Dán (`Ctrl+V`) hoặc kéo thả ảnh vào khung bên trái. "
            "**Click 2 điểm (góc trên-trái và góc dưới-phải)** trên ảnh gốc để cắt gọn 1 chiếc lá."
        )

        trang_thai_anh_goc = gr.State(value=None)
        trang_thai_diem_dau = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                khung_anh_goc = gr.Image(
                    type="pil",
                    sources=["upload", "clipboard"],
                    label="1. Ảnh gốc (Click 2 góc quanh chiếc lá để cắt)",
                )
                nut_hoan_tac = gr.Button(value="Hoàn tác về ảnh gốc")

            with gr.Column(scale=1):
                khung_anh_da_cat = gr.Image(
                    type="pil",
                    interactive=False,
                    label="2. Vùng lá đã cắt đưa vào mô hình",
                )

            with gr.Column(scale=1):
                hop_ket_qua = gr.Textbox(
                    label="3. Kết quả phân tích chi tiết",
                    lines=6,
                )
                nhan_xac_suat = gr.Label(
                    num_top_classes=3,
                    label="Xếp hạng 3 lớp có xác suất cao nhất",
                )

        khung_anh_goc.upload(
            fn=xu_ly_khi_tai_anh_moi,
            inputs=[khung_anh_goc],
            outputs=[
                trang_thai_anh_goc,
                trang_thai_diem_dau,
                khung_anh_da_cat,
                hop_ket_qua,
                nhan_xac_suat,
            ],
        )

        khung_anh_goc.select(
            fn=xu_ly_click_cat_anh,
            inputs=[trang_thai_anh_goc, trang_thai_diem_dau],
            outputs=[
                trang_thai_diem_dau,
                khung_anh_da_cat,
                hop_ket_qua,
                nhan_xac_suat,
            ],
        )

        nut_hoan_tac.click(
            fn=xu_ly_hoan_tac_anh_goc,
            inputs=[trang_thai_anh_goc],
            outputs=[
                trang_thai_diem_dau,
                khung_anh_da_cat,
                hop_ket_qua,
                nhan_xac_suat,
            ],
        )

    giao_dien.launch(inbrowser=True)


if __name__ == "__main__":
    khoi_chay_giao_dien()