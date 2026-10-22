# Strata sur RHEL / CentOS 7 : compiler et lancer Qwen3.6-35B-A3B

Toutes les commandes, dans l'ordre. Les explications sont dans [docs/RHEL7.md](docs/RHEL7.md).
Carte visée : GTX 10 (Pascal, `sm_61`). Pour une autre carte, changez `--arch` (`nvidia-smi --query-gpu=compute_cap --format=csv`, sans le point : 7.5 devient 75).

> **Vérifié** (conteneur CentOS 7.9, sans carte graphique) : la compilation de `strata` et `libstrata_kq.so`, avec CMake 3.20.2, sans pip ni ninja.
> **Vérifié aussi** : le serveur (414 tests) et l'extraction du tokenizer fonctionnent avec `jinja2` et `regex` absents (Python 3.13, sans aucun module).
> **Non vérifié** : tout ce qui touche à une carte ou à un vrai GGUF, c'est-à-dire les étapes 4 et 5, et Python 3.8 sur cette branche.

## 1. Prérequis (une fois)

```sh
# compilateur C++20 (RHEL 7 : activer d'abord le dépôt rhscl ; CentOS 7 : sudo yum install centos-release-scl)
sudo yum install devtoolset-11-gcc-c++ devtoolset-11-binutils git make

# CUDA 12.4 : dépôt NVIDIA RHEL 7 (/etc/yum.repos.d/cuda-rhel7.repo), puis
sudo yum install cuda-nvcc-12-4 cuda-cudart-devel-12-4 cuda-driver-devel-12-4 libcublas-devel-12-4 cuda-cccl-12-4
```

Dépôt NVIDIA :

```ini
[cuda-rhel7]
name=NVIDIA CUDA (RHEL 7)
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64
enabled=1
gpgcheck=0
```

Pilote NVIDIA : R525 ou plus récent. CMake : 3.20 ou plus (voir `cmake --version`). Sinon, sans root ni pip :

```sh
curl -LO https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2-linux-x86_64.tar.gz
tar xzf cmake-3.20.2-linux-x86_64.tar.gz -C ~
```

## 2. Sources

```sh
git clone https://github.com/Urkgor/Strata.git ~/Strata
cd ~/Strata && git checkout feature/RHEL7_compatibility

# llama.cpp EXACTEMENT à ce commit (un autre commit fait échouer le build : « no instance of function template mul_mat_q_case »)
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp && git checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
```

Sans accès à GitHub depuis la machine : téléchargez ailleurs `https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.tar.gz`, puis décompressez-le (le script ne peut alors pas vérifier la version).

## 3. Compiler

```sh
cd ~/Strata
tools/build_rhel7.sh --cmake ~/cmake-3.20.2-linux-x86_64/bin/cmake --arch 61 --ggml ~/llama.cpp
```

- Sans `--ggml`, CMake télécharge lui-même le bon llama.cpp (il faut GitHub).
- Sans `--cmake`, le script prend le premier `cmake` 3.20 ou plus du `PATH`.
- Moins de RAM : `--jobs 2`. Un `Killed` dans le journal veut dire « trop de tâches pour la mémoire ».
- Résultat : `build-rhel7/strata`. Journal complet : `build-rhel7/build.log` (le script affiche tout seul les premières erreurs).

Vérifier :

```sh
tools/check_glibc_symbols.sh build-rhel7/strata build-rhel7/libstrata_kq.so     # doit finir par « (ok) »
build-rhel7/strata --help | head
```

## 4. Tester le moteur seul avec un GGUF

```sh
G=/data/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf        # le premier morceau si le fichier est découpé

# quelques tokens (ids) en sortie
build-rhel7/strata --model $G --tokens "1,2,3" --n 16 --max-context 4096

# vitesse : prompt de 512 tokens, génération de 64 tokens
build-rhel7/strata --model $G --bench-pp 512 --bench-tg 64
```

Options utiles de ce chemin (toutes les options du moteur : `build-rhel7/strata --help`) : `--gpu 0`, `--max-context N`, `--expert-vram GIB`, `--cpu-threads N`, `--mtp N`, `--no-graph`.

## 5. Lancer le serveur (web + API OpenAI / Anthropic)

Il faut **Python 3.8 ou plus et aucun module** : le serveur et le tokenizer n'utilisent que la bibliothèque standard (ni `jinja2`, ni `regex`, ni `numpy`, ni pip). RHEL 7 a Python 3.6 : prenez Python 3.8 (`rh-python38` des Software Collections), ou le Python 3.12 autonome de `setup.sh` (`.python/python/bin/python3`).

```sh
S=~/Strata
G=/data/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
P=$HOME/strata-data/packs/qwen36

# 5a. tokenizer et modèle de conversation, une seule fois (extraits du GGUF, quelques secondes)
cd $S
python3 tools/strata_tokenizer.py --gguf $G --out $P

# 5b. configuration
cat > $HOME/strata-qwen36.json <<EOF
{
 "exe": "$S/build-rhel7/strata",
 "args": ["--native", "$G", "--max-context", "32768", "--prefill", "auto", "--mtp", "2"],
 "cwd": "$S",
 "tokenizer": "$P/tokenizer",
 "model_name": "qwen3.6-35b-a3b",
 "arch": "qwen35moe",
 "lib_dirs": ["/usr/local/cuda-12.4/lib64"],
 "log": "$HOME/strata-qwen36.log",
 "port": 8080
}
EOF

# 5c. serveur
python3 serve/server.py --engine strata --config $HOME/strata-qwen36.json --port 8080
```

- Interface web : `http://127.0.0.1:8080/`
- Un appel à l'API :

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":64}'
```

- Laissez l'écoute sur `127.0.0.1`. Pour d'autres machines : `--host 0.0.0.0` et une clé d'API (`"api_key"` dans le JSON).
- **GGUF sans couche MTP** (un GGUF « ordinaire » de Qwen3.6, pas ceux d'Unsloth nommés `-MTP-GGUF`) : le moteur répondait `strata-q36: missing qwen35moe.nextn_predict_layers`. Corrigé (`git pull`, puis relancer `tools/build_rhel7.sh`) : l'absence de cette clé veut dire « pas de bloc MTP ». Dans ce cas mettez `"--mtp", "0"` : les brouillons MTP n'existent pas pour ce fichier (le moteur les désactive de lui-même, mais autant ne pas les demander).
- Réglages à essayer : `--max-context` plus petit (16384) pour économiser la mémoire ; `"--mtp", "0"` si les brouillons posent problème ; `"--expert-vram", "6"` dans `args` pour limiter la VRAM (valeur d'essai, pas une mesure).

## Si ça casse

| Message | Cause et remède |
| --- | --- |
| `strata-q36: blk.N.ssm_alpha.weight is Q6_K, expected F32, BF16, F16 or Q8_0` | GGUF « K-quant ordinaire » (Q4_K_M…) qui stocke ces petits tenseurs quantifiés, moteur d'avant le correctif : `git pull` et recompiler (étape 3). Le moteur les élargit maintenant en f32 au chargement |
| `strata-q36: missing qwen35moe.nextn_predict_layers` | GGUF sans couche MTP, moteur d'avant le correctif : `git pull` et recompiler (étape 3) |
| `engine died` / `the engine exited before it was ready` (`server.py`) | le message du serveur ne dit que ça : **la cause est dans les lignes du moteur lui-même**, dans `~/strata-qwen36.log` (`tail -60`). Sans le journal : lancer le moteur seul, `build-rhel7/strata --model $G --tokens "1,2,3" --n 8 --max-context 4096 2>&1 \| tee ~/moteur.log`. Bibliothèque manquante : `ldd build-rhel7/strata \| grep "not found"` |
| `Error 2` seul | enveloppe de `make` ; lire `build-rhel7/build.log` ou relancer avec `--jobs 1` |
| `no instance of function template ... mul_mat_q_case` | mauvais commit de llama.cpp (étape 2) |
| `Target ... requires the language dialect "CUDA20"` | CMake trop ancien (< 3.20) ou `CMakeLists.txt` pas à jour : `git pull` |
| `unsupported GNU version` | nvcc plus ancien que 12.4 : utiliser devtoolset-11 (GCC 11) |
| `Killed` / `out of memory` | `--jobs 1` |
| `libcudart.so.12: cannot open shared object file` | `export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH` |
| `CUDA driver version is insufficient` | pilote plus ancien que R525 |
| `SyntaxError` ou `invalid syntax` au lancement de Python | Python trop ancien (3.6 de RHEL 7) : Python 3.8 ou plus (étape 5) |
| `no kernel image is available` | mauvais `--arch` pour la carte |
