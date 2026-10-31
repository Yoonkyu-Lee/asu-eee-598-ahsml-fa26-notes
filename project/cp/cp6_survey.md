# CP6 조사 — 홈 노드 박스 후보 엣지 칩: 인코더·암호·상시 전력

2026-10-31 작성. 모든 URL은 이 날 접속 기준.
우리는 아래 칩을 하나도 갖고 있지 않으므로 **이 문서의 숫자는 전부 출처 값**이다. 우리 측정(CP1–CP5)과 섞지 않는다.

표기 규칙
- **[V]** 벤더 주장 (데이터시트·제품 페이지·벤더 model zoo), **[I]** 독립 측정 (리뷰·논문·제3자), **[F]** 포럼·사용자 보고 (신뢰도 낮음), **[계산]** 출처 숫자로 우리가 나눈 값.
- TOPS는 sparse / dense를 구분. 표시가 없으면 벤더가 sparsity를 언급하지 않은 값이다.
- 찾지 못한 값은 "미확인".

---

## 1. 하드웨어 요약

| 칩 / 제품 | 최대 성능 (정밀도) | 전력: 일반 / 최대 | 유휴 전력 | 온칩 메모리 | 호스트 연결 | 가격 |
|---|---|---|---|---|---|---|
| **Hailo-8** (M.2, Pi AI HAT+ 26T) | 26 TOPS INT8, dense/sparse 언급 없음 [V] [hailo.ai/hailo-8](https://hailo.ai/hailo-8) | 2.5 W typical [V] [hailo.ai/hailo-8](https://hailo.ai/hailo-8); 모듈 최대 8.25 W, TDP 8.65 W, ResNet-50 915 FPS에서 3.3 W, MobileNet-SSD 996 FPS에서 2.4 W [V, 모듈 제조사 Geniatech Rev 2.1, 2022-11](https://file.geniatech.com/download/module/AIM-M-H8_Specification_V1.0_En.pdf) | 미확인 | 크기 미확인. "필요한 메모리를 모두 다이에 집적, 외부 메모리 불필요" [V] [hailo.ai/hailo-8](https://hailo.ai/hailo-8) | PCIe Gen3 x4 (M key), x2 (B+M, A+E) [V] [Hailo M.2 페이지](https://hailo.ai/products/ai-accelerators/hailo-8-m2-ai-acceleration-module/) | AI HAT+ 26 TOPS $110 (2024-10 출시가) [I] [LinuxGizmos](https://linuxgizmos.com/raspberry-pi-expands-ai-line-with-ai-hat-offering-up-to-26-tops/) |
| **Hailo-8L** (Pi AI Kit, AI HAT+ 13T) | 13 TOPS INT8 [V] [Hailo-8L M.2 데이터시트 Rev 4.0, 2025-11](https://hailo.ai/hailo-files/hailo-8l-m-2-key-a-e-et-datasheet-en/) | ResNet-50 433 FPS에서 1.9 W, MobileNet-SSD 500 FPS에서 1.4 W; 최대 6.6 W (= TDP) [V] [같은 데이터시트](https://hailo.ai/hailo-files/hailo-8l-m-2-key-a-e-et-datasheet-en/) | 미확인 | 미확인 (외부 DRAM 없음, 위 Hailo-8과 같은 계열) | PCIe Gen3 x2 [V] [같은 데이터시트](https://hailo.ai/hailo-files/hailo-8l-m-2-key-a-e-et-datasheet-en/) | AI Kit / AI HAT+ 13 TOPS $70 [V] [raspberrypi.com/products/ai-hat](https://www.raspberrypi.com/products/ai-hat/) |
| **Hailo-10H** (Pi AI HAT+ 2) | 40 TOPS INT4 [V] [raspberrypi.com/products/ai-hat-plus-2](https://www.raspberrypi.com/products/ai-hat-plus-2/); INT8 20 TOPS [I, heise가 Hailo 인용, 2026-01-15](https://heise.de/-11136612) | 2.5 W (Hailo 수치, heise 인용) [I] [heise](https://heise.de/-11136612) | 미확인 | 온보드 LPDDR4X 8 GB [V] [raspberrypi.com](https://www.raspberrypi.com/products/ai-hat-plus-2/) | Pi 5 PCIe (세대 미확인) | $200 (2026-10-31 공식 페이지) [V] [raspberrypi.com](https://www.raspberrypi.com/products/ai-hat-plus-2/); 출시가 $130 [I] [heise](https://heise.de/-11136612) |
| **Google Coral Edge TPU** (USB, M.2) | 4 TOPS INT8, 2 TOPS/W [V] [M.2 데이터시트 v1.5](https://coral.ai/static/files/Coral-M2-datasheet.pdf) | M.2: MobileNet v2 416 FPS에서 1.4 W (500 MHz), 250 MHz에서 0.9 W; 평균 전류 < 500 mA, 순간 ~3 A @3.3 V [V] [M.2 데이터시트 v1.5](https://coral.ai/static/files/Coral-M2-datasheet.pdf). USB: 피크 900 mA @5 V [V] [USB 데이터시트 v1.4](https://coral.ai/static/files/Coral-USB-Accelerator-datasheet.pdf) (= 4.5 W [계산]) | M.2: 375–400 mW [V] [M.2 데이터시트](https://coral.ai/static/files/Coral-M2-datasheet.pdf) | 파라미터 캐시용 SRAM 약 8 MB [V] [Edge TPU Compiler 문서](https://coral.ai/docs/edgetpu/compiler/) | M.2: PCIe Gen2 x1; USB 3.0 (5 Gbps) [V] [데이터시트](https://coral.ai/static/files/Coral-M2-datasheet.pdf) | USB $59.99, M.2 $24.99, Dual Edge TPU M.2 $39.99 [V] [coral.ai/products](https://coral.ai/products/) |
| **Jetson Orin Nano 8GB (Super)** | 67 TOPS **sparse** / 33 TOPS **dense** INT8 (GPU만, DLA 없음); FP16 17 TFLOPs [V] [NVIDIA 블로그, JetPack 6.2](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) | 전력 모드 15 W / 25 W / MAXN SUPER (원래 7 W / 15 W) [V] [같은 블로그](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) | 개발 키트 4.7 W (7 W 모드), 디스플레이 끄고 2.5 W [F] [NVIDIA 포럼](https://forums.developer.nvidia.com/t/reducing-idle-power-on-orin-nano-super-dev-kit/358482) | GPU L2 미확인. DRAM 102 GB/s [V] [블로그](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) | SoC (CPU 6× A78 1.7 GHz 일체) | 개발 키트 $249 (2024-12 출시가) [I] [Phoronix](https://www.phoronix.com/news/NVIDIA-Jetson-Orin-Nano-Super) |
| **Jetson Orin NX 16GB (Super)** | 157 sparse / 78 dense INT8 (GPU 77/38 + DLA 80/40) [V] [블로그](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) | 10 / 15 / 25 / 40 W / MAXN SUPER [V] 같은 출처 | 미확인 | 미확인. DRAM 102 GB/s [V] | SoC | 미확인 |
| **Raspberry Pi 5 CPU 단독** | 가속기 없음. 4× Cortex-A76 2.4 GHz [V] [raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-5/) | stress-ng 8.9 W (2GB, D0) / 9.8 W (4GB·8GB, C1) [I] [Jeff Geerling, 2024-08-29](https://www.jeffgeerling.com/blog/2024/new-2gb-pi-5-has-33-smaller-die-30-idle-power-savings) | 2.4 W (2GB) / 3.3 W (4GB) / 3.2 W (8GB) [I] 같은 출처 | 해당 없음 | PCIe 2.0 x1 (가속기용) [V] [raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-5/) | 16GB $305 (2026-10-31 공식 페이지). 다른 용량은 미확인 |
| **Rockchip RK3588 NPU** | 6 TOPS INT8; int4/int8/int16/FP16/BF16/TF32 지원 [V] [Rockchip RK3588 Brief, 2022](https://github.com/dora-rs/dora-hardware/files/11902112/RK3588.Brief.Datasheet.pdf) | 미확인 (보드 단위 신뢰할 측정 못 찾음) | 미확인 (Rock 5B 약 4 W라는 포럼 보고만 있음 [F] [Radxa 포럼](https://forum.radxa.com/t/introduce-rock-5b-arm-desktop-level-sbc/8361?page=43)) | 블록도에 NPU 옆 "1MB Share Memory" [V] [Brief](https://github.com/dora-rs/dora-hardware/files/11902112/RK3588.Brief.Datasheet.pdf). NPU 전용인지 미확인 | SoC (4× A76 + 4× A55) | 미확인 (보드마다 다름) |
| (참고) Intel Meteor Lake NPU, 우리 노트북 | INT8 9.5 TOPS @1.16 GHz 측정 클럭 / Intel 발표 11.5 TOPS @1.4 GHz [I] [Chips and Cheese, 2024-04](https://chipsandcheese.com/p/intel-meteor-lakes-npu) | 칩 전체 16.2 W (우리 CP5 측정, 헤드별 마스크 우회 후) | CP5 참조 | 타일당 2 MB SRAM × 2 [I] [Chips and Cheese](https://chipsandcheese.com/p/intel-meteor-lakes-npu) | SoC 내장 | 노트북 일체 |

메모
- Raspberry Pi AI HAT+는 PCIe Gen 3 모드로 전환된다고 Raspberry Pi가 발표했다 [I] [Elektroniknet](https://www.elektroniknet.de/international/raspberry-pi-releases-ai-hat-with-up-to-26-tops.221554.html). Pi 5 기본 규격은 PCIe 2.0 x1이다. 즉 Pi 위의 Hailo는 **x1 링크**이고, 아래 Hailo model zoo 숫자(x86 호스트, PCIe Gen3 x4)보다 느릴 수 있다 (Pi에서 측정된 MiniLM 수치는 미확인).
- Coral USB는 최대 클럭 런타임을 고르면 속도와 전력이 함께 오른다(최대 클럭 = 감소 클럭의 2배) [V] [USB 데이터시트 v1.4](https://coral.ai/static/files/Coral-USB-Accelerator-datasheet.pdf).
- Jetson의 67/33 TOPS는 GPU Tensor Core 수치다. 아래 3절처럼 ZK 연산은 Tensor Core를 쓰지 않는다.

---

## 2. Transformer(인코더) 지원

| 칩 | 툴체인 | BERT류 인코더 / softmax·LayerNorm·GELU | 가변 길이 | 배치 | 공개 벤치마크 (조건) |
|---|---|---|---|---|---|
| **Hailo-8** | Hailo Dataflow Compiler (DFC) v2.20.0 | **all-MiniLM-L6-v2가 공식 model zoo에 있다** (ViT, CLIP 텍스트 인코더, SigLIP 텍스트 인코더도) [V] [Hailo model zoo, HAILO8 sentence embedding, v2.20.0](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.20.0/docs/public_models/HAILO8/HAILO8_sentence_embedding_generation.rst). 레이어 정밀도는 혼합: a8_w8_a8 121개, a16_w16_a16 65개 등 [V] [프로파일러 보고서](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/all_minilm_l6_v2_profiler_results_compiled_runtime_data.html). 2025-11 시점에는 Hailo 직원이 "BERT would remain out of scope for the time being"이라고 답했다 [F] [Hailo 포럼, 2025-11-03](https://community.hailo.ai/t/raspberry-pi-ai-hat-disappointment/18099). 그 후 바뀐 것이다 | **고정 128토큰** (입력 16×8×384, 384차원 임베딩을 입력으로 받음 → 토큰 임베딩 조회는 호스트) [V] [model zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.20.0/docs/public_models/HAILO8/HAILO8_sentence_embedding_generation.rst), [CHANGELOG "128-token input"](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/CHANGELOG.rst). reshape 출력 모양은 컴파일 시점에 정적이어야 한다 [F] [Hailo 포럼](https://community.hailo.ai/t/reshape-layer-error/13404) | 1, 8 | MiniLM-L6: **156 FPS (배치 1), 742 FPS (배치 8)**, 프로파일러 지연 6.37 ms, **3 contexts** (한 번에 칩에 다 안 올라감). 호스트 i5-9400, PCIe Gen3 x4. 품질 Retrieval@10 61.7 (float) → 61.6 (HW), MTEB ArguAna [V] [model zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.20.0/docs/public_models/HAILO8/HAILO8_sentence_embedding_generation.rst), [프로파일러](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/all_minilm_l6_v2_profiler_results_compiled_runtime_data.html). CLIP ViT-B/32 텍스트 인코더(77토큰, 37.8 M 파라미터): 35.5 / 153 FPS [V] [model zoo text-image retrieval](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.20.0/docs/public_models/HAILO8/HAILO8_text_image_retrieval.rst) |
| **Hailo-8L** | 같음 | 같은 MiniLM 모델 [V] [HAILO8L 표](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.20.0/docs/public_models/HAILO8L/HAILO8L_sentence_embedding_generation.rst) | 고정 128토큰 | 1, 8 | MiniLM-L6: **121 FPS (배치 1), 492 FPS (배치 8)**, 프로파일러 지연 8.21 ms, **5 contexts**. 같은 x86 호스트 조건 [V] [같은 표](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.20.0/docs/public_models/HAILO8L/HAILO8L_sentence_embedding_generation.rst), [프로파일러](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8l/all_minilm_l6_v2_profiler_results_compiled_runtime_data.html) |
| **Hailo-10H** | DFC v5.5.0 | MiniLM 공식 지원 (v5.5.0 CHANGELOG에 "First sentence-embedding model") [V] [CHANGELOG](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/CHANGELOG.rst) | 고정 128토큰 | 1, 8 | MiniLM-L6: 249 FPS (배치 1), 877 FPS (배치 8), x86 호스트 PCIe Gen3 x4 [V] [HAILO10H 표](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_sentence_embedding_generation.rst) |
| **Coral Edge TPU** | TFLite + Edge TPU Compiler | 사실상 불가. 지원 연산표에 LayerNorm, GELU, BatchMatMul이 없고, Softmax는 "1-D 입력, 최대 16,000원소"만 [V] [models-intro](https://coral.ai/docs/edgetpu/models-intro/). 표에 없는 연산이 있으면 "일부만 컴파일"되고 나머지는 CPU로 간다 [V] [compiler 문서](https://coral.ai/docs/edgetpu/compiler/) | **정적만**: "Tensor sizes are constant at compile-time (no dynamic sizes)" [V] [models-intro](https://coral.ai/docs/edgetpu/models-intro/) | 고정 | 공식 벤치마크는 비전 모델 20개뿐, transformer/BERT 없음 [V] [benchmarks](https://coral.ai/docs/edgetpu/benchmarks/). BERT/MiniLM 수치 미확인 |
| **Jetson Orin** | TensorRT (JetPack) | BERT 공식 지원 (MLPerf 제출) | **동적 가능**: 빌드 시 min/opt/max 모양의 optimization profile을 정의, opt 모양 기준으로 커널 튜닝 [V] [Torch-TensorRT 문서](https://docs.pytorch.org/TensorRT/contributors/dynamic_shapes.html), [TensorRT 개발 가이드 (DRIVE OS 판)](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/work-dynamic-shapes.html) | 자유 | MLPerf Inference v3.1 Edge, BERT (BERT-Large, SQuAD v1.1, seq 384 [정의 출처: Qualcomm MLPerf 자료](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Cloud%20AI%20100%20MLPerf%202.0%20inference%20performance%202022-05-16.pdf)): AGX Orin 단일 스트림 5.71 ms, offline 553.69 samples/s; **Orin NX 16GB offline 194.5 samples/s; Orin NX MaxQ 136.59 samples/s @ 시스템 17.04 W** (JetPack 5.1.1, TRT 8.5.2) [V] [Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks) → 0.125 J/sample [계산]. **Orin Nano BERT/MiniLM 수치 미확인.** ViT-base-patch16-224 FP16: Orin Nano 8GB Super 158 FPS [V] [JetPack 6.2 블로그](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) |
| **RK3588** | RKNN-Toolkit2, rknn_model_zoo v2.5.0 | transformer 예제 있음 (lite_transformer, CLIP 텍스트, Whisper, MMS-TTS). BERT/MiniLM 예제 미확인 [V] [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) | **미리 정한 모양 집합**: `rknn.config(dynamic_input=[[shape1],[shape2],...])` [V] [dynamic_shape 예제](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/dynamic_shape) | 미확인 | NPU 코어 1개, FP16, 최대 NPU 클럭, 전·후처리 제외: CLIP 텍스트 인코더 [1, 20] **96.1 FPS**; lite-transformer encoder (임베딩 256, 16토큰) 986.9 FPS [V] [rknn_model_zoo README](https://github.com/airockchip/rknn_model_zoo) |
| **Pi 5 CPU** | ONNX Runtime, PyTorch 등 범용 | 제한 없음 | 자유 | 자유 | Pi 5에서 MiniLM 지연 공개값 **미확인** |

메모
- Hailo 숫자는 모두 **x86 호스트 + PCIe Gen3 x4**에서 잰 벤더 값이다. Hailo-8은 외부 DRAM이 없어서 MiniLM이 3개(8L은 5개) context로 쪼개진다. context 사이에 무엇을 다시 싣는지는 문서로 확인 못 했지만, Pi 5의 x1 링크에서는 이 전환 비용이 더 커질 수 있다 (추정, 미측정).
- Hailo-8L 에너지 [계산, 추정]: MiniLM 전력은 공개되지 않았다 (프로파일러 "power": "N/A"). 비전 모델 기준 1.4–1.9 W를 빌려 쓰면 배치 1에서 약 12–16 mJ/임베딩, 배치 8에서 약 3–4 mJ다. 비교: 우리 노트북 NPU는 칩 전체 0.13 J/게시물 (CP5), 분석 모델은 8.6 mJ (CP4, 45 nm 동적만).
- CLIP 텍스트 인코더(RK3588, Hailo)는 MiniLM보다 큰 모델이다 (Hailo 표 기준 37.8 M vs 10.6 M 파라미터). 그래서 RK3588의 MiniLM 속도는 이보다 빠를 가능성이 높지만 공개값은 없다.

---

## 3. 암호 가속 (큰 정수 모듈러 연산: 페어링, MSM, NTT, Ed25519)

| 칩 | 큰 정수 연산 가능? | 근거 |
|---|---|---|
| Hailo-8 / 8L / 10H | **불가** (신경망 데이터플로 전용) | 정밀도 모드가 a8_w8 / a16_w16 같은 8·16비트 [V] [Hailo 프로파일러](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/all_minilm_l6_v2_profiler_results_compiled_runtime_data.html); 10H는 INT4/INT8 [I] [heise](https://heise.de/-11136612). 254–381비트 모듈러 곱셈 경로 없음. 범용 연산 프로그래밍 수단 공개 안 됨 (미확인) |
| Coral Edge TPU | **불가** | 텐서 파라미터는 "8-bit fixed-point numbers; int8 or uint8" [V] [models-intro](https://coral.ai/docs/edgetpu/models-intro/) |
| RK3588 NPU | **불가** (NPU). SoC 암호 블록은 "Crypto (RSA 2K, HASH 256…)"만 표기, ECC·페어링 언급 없음 | [V] [Rockchip Brief](https://github.com/dora-rs/dora-hardware/files/11902112/RK3588.Brief.Datasheet.pdf) |
| Pi 5 CPU (Cortex-A76) | CPU 소프트웨어로만. 전용 가속 없음 | arkworks `ark-ff`의 인라인 어셈블리 최적화는 x86_64에만 있고, arm64 저사양 기기(Pi급)에서 Groth16 검증이 "very slow"였다는 사용자 보고 (수치 없음) [F] [arkworks issue #805, 2024-03](https://github.com/arkworks-rs/algebra/issues/805) |
| Jetson Orin (GPU) | **CUDA로 가능 (이론상)**. 하지만 Jetson에서 측정된 ZK 수치는 **미확인** | GPU ZK 연산은 "exclusively on the GPU's 32-bit integer pipeline"이고 Tensor Core·FP 유닛은 놀아서 [I] [ZKProphet, IISWC 2025](https://arxiv.org/abs/2509.22684), **광고된 67/33 TOPS는 ZK와 무관**하다. 이 논문은 V100–H100 데이터센터 GPU만 평가, Jetson 없음 |

Jetson용 GPU ZK 라이브러리 현황
- **ICICLE (Ingonyama)**: BN254, BLS12-381 MSM·NTT 지원, CPU 백엔드는 MIT [V] [GitHub](https://github.com/ingonyama-zk/icicle). CUDA 백엔드는 "closed-source components that require a valid license", 기본은 R&D 무료 라이선스, 상용은 영업 문의 [V] [설치 문서](https://dev.ingonyama.com/start/architecture/install_gpu_backend). aarch64 / Jetson 지원 언급 **없음** (미확인).
- **snarkrs**: Groth16 (BN254), CPU / Metal / CUDA / WebGPU 백엔드. CUDA 수치는 AWS g4dn.2xlarge (T4) 기준, 예: 2만 제약 회로 warm proving 22.8 ms (CUDA) vs 117.7 ms (CPU) [V/I, 프로젝트 자체 측정] [GitHub](https://github.com/zemse/snarkrs). Jetson 언급 없음.
- **Jetson/Orin에서 ZK (MSM, NTT, Groth16) 측정 논문·벤치마크: 미확인.** 여러 검색에서 하나도 못 찾았다.
- 참고: BLS12-381 페어링 전용 저전력 IoT 암호 프로세서 연구가 있다 ("two orders of magnitude energy savings", 수치는 초록에 없음) [I] [arXiv 2201.07496](https://arxiv.org/abs/2201.07496). ML NPU와 한 칩에 묶은 사례는 미확인.

---

## 4. 상시 가동 가정용 기기 전력 (맥락)

| 기기 | 유휴 | 동작 | 출처 |
|---|---|---|---|
| Wi-Fi 5 라우터 Netgear R7500 | 8.3 W | 9.5 W (무선 1대 부하) | [I] [Legit Reviews, 2015-04-02, 벽 콘센트 Kill-A-Watt](https://legitreviews.com/netgear-nighthawk-x4-ac2350-wifi-router-review_160827/7) |
| Wi-Fi 7 메시 ASUS ZenWiFi BQ16 | 약 18 W | 약 20 W | [I] [Guru3D](https://www.guru3d.com/review/asus-zenwifi-bq16-review-quad-band-mesh-with-wifi-7/page-9/). 원문 접근 403, 검색 요약으로만 확인 → 재확인 필요 |
| 2베이 NAS Synology DS224+ (Celeron J4125) | 4.41 W (HDD 최대 절전) | 14.69 W (Access) | [V] [Synology 제품 사양, 2026-08-06 갱신](https://global.download.synology.com/download/Document/Hardware/ProductSpec/DiskStation/24-year/DS224+/enu/Product_Spec_DS224+_enu.pdf) |
| Raspberry Pi 5 | 2.4–3.3 W | stress-ng 8.9–9.8 W | [I] [Jeff Geerling, 2024-08-29](https://www.jeffgeerling.com/blog/2024/new-2gb-pi-5-has-33-smaller-die-30-idle-power-savings) |
| Jetson Orin Nano Super 개발 키트 | 4.7 W (7 W 모드), 디스플레이 끄면 2.5 W | 활성 6.2 W (7 W 모드) | [F] [NVIDIA 포럼](https://forums.developer.nvidia.com/t/reducing-idle-power-on-orin-nano-super-dev-kit/358482) |

규제
- **EU Regulation (EU) 2023/826** (2025-05-09부터 적용): networked standby 한도 **HiNA(라우터·스위치·AP·모뎀 등 높은 네트워크 가용성 기기) 8 W → 2027년 7 W, 비-HiNA 2 W** [I] [AVIXA 해설](https://www.avixa.org/explore/articles/eu-networked-standby-rules). EU 집행위 페이지: 2027년부터 networked standby "2 to 7 Watts depending on the product", standby 0.5 W, off 0.3 W [V] [EC energy-efficient products](https://energy-efficient-products.ec.europa.eu/product-list/standby-networked-standby-and-mode_en). EUR-Lex 원문 부속서는 접근 실패 → 원문 확인은 미확인.
  - 해석 (우리 판단): 이 한도는 **대기(standby) 모드**에 걸린다. 수집·검색을 계속하는 박스는 활성 모드이므로 이 규정이 직접 상한을 주지는 않는다. 다만 "항상 연결된 네트워크 장비는 대기 때 7–8 W 이하"라는 규제 쪽 기대치로 쓸 수 있다.
- 미국: 연방 규제 대신 Small Network Equipment 자율 협약 (2016년 판매 기기의 98.3%가 기준 충족) [V, 업계 보고서] [SNE VA 2016 Annual Report](https://www.technicolor.com/sites/default/files/medialib/document/csr/Small%20Network%20Equipment%20-%20US%20Voluntary%20Agreement%20-%202016%20Annual%20Report.pdf). 기기별 허용 W 값은 미확인.

---

## 5. 공개된 ZK·서명 CPU 비용 (우리 측정 대조용)

| 연산 | 시간 | 하드웨어 / 라이브러리 / 조건 | 출처 |
|---|---|---|---|
| BN254 페어링 1회 | 0.49 ms (gnark-crypto), 0.48 ms (mcl), 1.07 ms (arkworks+asm) | AWS z1d.3xlarge Xeon 3.4 GHz, 터보 끔, 2021-01 | [I] [gnark eccbench](https://hackmd.io/@gnark/eccbench) |
| BLS12-381 페어링 1회 | 0.61 ms (blst), 0.71 ms (gnark-crypto), 1.62 ms (arkworks) | 같은 조건 | [I] 같은 출처 |
| **Groth16 검증 (RLN-v3 회로)** | **1.7 ms** (증명 생성 90 ms) | MacBook Air M2 16 GB, gnark, BN254, 2024-05-13 | [V, 프로젝트 자체] [Vac/Logos RLN-v3](https://research.logos.co/rlog/rln-v3) |
| **Semaphore v4 검증** | **8.74–8.89 ms** (그룹 1–2000명) | MacBook Pro M2 Pro, Node.js 23.10.0, snarkjs, 트리 깊이 20 | [V] [Semaphore benchmarks](https://docs.semaphore.pse.dev/benchmarks) (수치는 페이지의 이미지 표) |
| **Semaphore v4 증명 생성** | 165.8 ms (1명) – 310.7 ms (2000명); v3는 약 322–327 ms | 같은 조건. v4 회로 비선형 제약 6,431개 | [V] 같은 출처 |
| RLN (v1, Waku) | "sub-second" 증명 생성, 수치는 그래프에만 | AMD EPYC 7502P, 2023-11 | [V] [Vac/Logos RLN](https://research.logos.co/rlog/rln-anonymous-dos-prevention) |
| **BBS (BLS12-381) 증명 검증** | **1.643 ms** (SHA-256), 1.631 ms (SHAKE-256); 서명 검증 1.299 ms; 증명 생성 1.190 ms | Lenovo ThinkPad T14s Gen1, clang, 메시지 2개(64 B) 중 1개 공개, 1000회 평균, 워밍업 없음 | [V, 프로젝트 자체] [Fraunhofer AISEC libbbs](https://github.com/Fraunhofer-AISEC/libbbs) |
| Groth16 검증 on ARM SBC (Pi 5 등) | **미확인** | arm64 저사양에서 느리다는 정성 보고만 | [F] [arkworks #805](https://github.com/arkworks-rs/algebra/issues/805) |

메모
- 같은 Groth16 검증도 구현에 따라 약 5배 차이: gnark 1.7 ms (M2) vs snarkjs 8.8 ms (M2 Pro, JS). 하드웨어도 달라 직접 비교는 아니다.
- Groth16 검증은 페어링 몇 번이 대부분이라 회로 크기에 거의 무관하다. 1.7 ms면 코어당 약 590회/s [계산].
- 우리 측정과 맞출 때: (1) 라이브러리(gnark / arkworks / snarkjs / rapidsnark), (2) x86 어셈블리 경로 유무, (3) 배치 검증 사용 여부를 같이 적어야 한다.

---

## 우리 논지에 대한 함의

- **인코더 처리량만으로는 기존 엣지 칩이 충분해 보인다 (CP5와 같은 결론).** Hailo-8L이 MiniLM-L6을 배치 1에서 121 /s, 배치 8에서 492 /s로 돌린다 (벤더, x86 호스트). 모듈 전력은 1.4–1.9 W급이다 (비전 모델 기준). 우리 노트북 NPU를 고친 값(8.15 ms, 칩 전체 16.2 W)과 지연은 비슷하다 (Hailo-8L 프로파일러 8.21 ms). 전력은 약 8–12배 작다 (16.2 W ÷ 1.4–1.9 W [계산]). 단 칩 전체 측정과 모듈 벤더 값이라 측정 범위가 다르다. "인코더가 느려서 우리 칩이 필요하다"는 주장은 이 근거로는 서지 않는다.
- **약점은 모양(shape)이다.** Hailo는 128토큰 고정, Coral은 동적 크기 불가에 LayerNorm·BatchMatMul 미지원, RKNN은 미리 정한 모양 집합만 받는다. 동적 모양을 제대로 받는 것은 Jetson(TensorRT 프로파일)뿐이다. 256토큰 게시물과 길이가 들쭉날쭉한 입력(패딩 낭비, 긴 글 잘라 쪼개기)에 대한 공개 수치는 없다. Hailo-8은 외부 DRAM이 없어 MiniLM을 3–5 context로 쪼갠다. CP5에서 본 "짧은 게시물마다 가중치를 다시 싣는다"는 문제와 같은 자리일 수 있다 (미측정 추정).
- **암호 쪽이 가장 확실한 빈자리다.** 조사한 NPU(Hailo, Coral, RK3588)는 모두 8/16비트 MAC 배열이라 페어링·MSM을 못 한다. Jetson GPU는 CUDA로 가능하지만, ZK 연산은 INT32 파이프라인을 쓰므로 광고 TOPS와 무관하다. Jetson에서 측정된 ZK 수치는 하나도 못 찾았고, 대표 라이브러리(ICICLE) CUDA 백엔드는 유료 라이선스다. 결국 박스에서 증명 검증은 ARM CPU가 맡는다. 그런데 ARM SBC에서의 Groth16·BBS 검증 공개값이 없다. 스팸 폭주 때 병목이 인코더가 아니라 검증일 가능성을 **우리가 Pi급 ARM에서 직접 재야** 한다. "ML + crypto 한 칩"이 논지의 가장 강한 후보다.
- **상시 전력은 기존 조합도 이미 가정용 범위다.** Pi 5 유휴 2.4–3.3 W에 Coral M.2 유휴 0.375–0.4 W를 더해도, 라우터 유휴 8.3 W나 NAS 최대 절전 4.41 W와 비슷하거나 낮다. 그래서 우리 하드웨어의 전력 이점은 유휴 W가 아니라 **게시물당 J, 검증당 J, 그리고 폭주 시 지속 W**로 보여야 한다. 박스 전체 유휴는 호스트 SoC·DRAM이 지배할 것으로 보인다 (추정).
- **Jetson Orin Nano는 "다 되는" 선택지지만 비싸다.** $249에 동적 모양과 CUDA 암호가 가능하다. 대신 TOPS는 sparse 67 / dense 33이고, 전력 모드는 7–25 W, 유휴는 포럼 보고 2.5–4.7 W다. 비교 대상으로는 "Pi 5 + Hailo-8L (약 $70 + Pi)"과 "Orin Nano"의 두 끝점을 잡으면 된다.
- **증거 공백 (다음 CP 할 일 후보):** Pi 5에서 Hailo-8L로 MiniLM을 돌린 실측(링크 x1, 전력 포함), ARM에서 Groth16·BBS 검증 실측, RK3588·Jetson Orin Nano의 MiniLM 실측. 모두 공개값이 없어서 비교표에 들어갈 수 없다.
