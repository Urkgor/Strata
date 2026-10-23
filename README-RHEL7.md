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
 "args": ["--native", "$G", "--max-context", "32768", "--kv", "f16", "--prefill", "auto", "--mtp", "2"],
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

- Par défaut le serveur n'écoute que sur `127.0.0.1` : **aucun autre poste ne peut s'y connecter**. Voir « Se connecter depuis un autre poste » ci-dessous.
- **GGUF sans couche MTP** (un GGUF « ordinaire » de Qwen3.6, pas ceux d'Unsloth nommés `-MTP-GGUF`) : le moteur répondait `strata-q36: missing qwen35moe.nextn_predict_layers`. Corrigé (`git pull`, puis relancer `tools/build_rhel7.sh`) : l'absence de cette clé veut dire « pas de bloc MTP ». Dans ce cas mettez `"--mtp", "0"` : les brouillons MTP n'existent pas pour ce fichier (le moteur les désactive de lui-même, mais autant ne pas les demander).
- **`"--kv", "f16"`** : le cache de contexte en demi-précision (2 octets au lieu de 4 par valeur) : moitié de VRAM (45 Ko → 22 Ko par token, soit 128 000 tokens ≈ 2,9 Go au lieu de 5,9 Go) et moitié de lecture mémoire en décodage long. Les valeurs sont arrondies en fp16 (comme le fait llama.cpp par défaut) : le texte peut différer très légèrement de celui du cache en f32, qui reste le comportement sans cette option (`"--kv", "f32"`). Avec une carte de 11 Go, essayez `--max-context 131072 --kv f16` ; la ligne du journal `context N cells (K / V cache in f16) … GiB of VRAM free` dit ce qui reste, puis `hybrid experts: N of 256 per layer in VRAM` combien d'experts tiennent. Pas mesuré sur une carte.
- Réglages à essayer : `--max-context` plus petit (16384) pour économiser la mémoire ; `"--mtp", "0"` si les brouillons posent problème ; `"--expert-vram", "6"` dans `args` pour limiter la VRAM (valeur d'essai, pas une mesure).

## Deux cartes : le contexte sur la 2e carte (`--kv-gpu`)

Idée : la carte de 11 Go garde les poids et le plus d'experts possible ; la carte de 8 Go porte le **cache de contexte et le calcul de l'attention**. Les experts ne bougent pas ; le calcul de l'attention est le même (mêmes noyaux, même ordre des opérations) : le texte doit être celui d'une seule carte, à une différence près possible si la 2e carte est d'une autre génération (cuBLAS choisit ses algorithmes par carte, pour le prompt). Option **désactivée par défaut** : sans `--kv-gpu` le moteur est inchangé.

```json
 "args": ["--gpu", "0,1", "--native", "$G", "--max-context", "131072", "--kv", "f16", "--kv-gpu", "1",
          "--prefill", "auto", "--mtp", "0"],
```

- **`"--gpu", "0,1"` dans `args`** (numéros de `nvidia-smi`) : la **première** carte listée garde les poids et les experts (mettez la 1080 Ti en premier), `--kv-gpu 1` désigne la **deuxième** du lot (pas le numéro de nvidia-smi). N'utilisez pas la clé `"gpu"` du fichier de configuration : le serveur ajouterait `--layer-split`, qui n'est pas ce qu'on veut ici.
- **Architecture des deux cartes** : le binaire doit contenir des noyaux pour la 2e carte. `tools/build_rhel7.sh` lit les cartes de `nvidia-smi` (toutes) ; si votre binaire a été compilé avant d'avoir branché/vu la 2e carte, relancez-le (`--arch "61;75"` à la main si besoin ; `nvidia-smi --query-gpu=name,compute_cap --format=csv`). Sinon le moteur s'arrête tout de suite avec `this build has no kernels for that card`.
- `--kv-gpu 0` (la carte principale) fait passer le même chemin de code sur une seule carte : utile pour vérifier que le texte (température 0) est identique à celui sans l'option.
- Mémoire sur la 2e carte : cache f16 à 128 000 tokens ≈ 2,9 Go (f32 : 5,9 Go) + tampons de l'attention ≈ 1 Go. Le journal dit `K / V cache and attention on device 1: … GiB of its VRAM free` ; si ça ne tient pas, le moteur le dit (`not enough VRAM on device 1 …`) et propose `--kv f16` ou un contexte plus court.
- Les deux cartes échangent, par couche d'attention (10 sur 40) et par token, 36 Ko aller + 16 Ko retour, via la mémoire de l'hôte (pas besoin de P2P entre les cartes). Pas de graphes CUDA avec cette option (un graphe ne peut pas couvrir deux cartes ; en mode hybride ils étaient déjà coupés). Le prompt envoie environ 52 Ko par token et par couche d'attention.
- **Pas testé sur une carte** (je n'ai pas de GPU ici) : le code compile sous CentOS 7 / CUDA 12.4 mais n'a jamais tourné. Dites-moi ce que donnent les lignes du journal, le débit en prompt et en décodage, avec et sans `--kv-gpu`.

## Décodage lent avec un long contexte (6 tk/s à 25 000 tokens contre 60 à petit contexte)

Cause trouvée dans le code : l'attention du décode lisait la mémoire des clés (K) de façon non coalescée (chaque fil lisait sa propre ligne de 1 Ko) et relisait K et V pour chacune des 8 têtes d'un même groupe. Corrigé (`git pull`, puis recompiler, étape 3) : un bloc sert les 8 têtes et lit K par tranches en mémoire partagée. Les résultats sont **identiques bit à bit** à l'ancien noyau (test CPU `q36_attn_gqa_test`), seul l'ordre des lectures change.
**Le gain n'est pas mesuré** (pas de carte ici) : comparez sur le même prompt de 25 000 tokens, avec et sans `STRATA_Q36_OLD_ATTN=1` (variable d'environnement : l'ancien noyau). Les deux doivent donner le même texte en décodage glouton (température 0).

## Se connecter depuis un autre poste (autre outil, autre machine)

```sh
# sur le serveur : écouter sur le réseau, avec une clé d'API
python3 serve/server.py --engine strata --config $HOME/strata-qwen36.json --port 8080 --host 0.0.0.0 --api-key UNE-CLE-LONGUE-ET-SECRETE

# sur le serveur : le pare-feu de RHEL 7 (firewalld) bloque le port par défaut
sudo firewall-cmd --permanent --add-port=8080/tcp && sudo firewall-cmd --reload
ss -ltn | grep 8080                        # doit montrer 0.0.0.0:8080 (et non 127.0.0.1:8080)
```

Le serveur affiche au démarrage `from other devices: http://<ip>:8080/   (API: http://<ip>:8080/v1)`. Réglages de l'outil qui se connecte :

| Réglage | Valeur |
| --- | --- |
| URL de base (OpenAI) | `http://<ip-du-serveur>:8080/v1` (avec `/v1`) |
| Clé d'API | la clé donnée à `--api-key` (en-tête `Authorization: Bearer …` ou `x-api-key`) |
| Modèle | `qwen3.6-35b-a3b` (le `model_name` du JSON) |
| API Anthropic (Claude Code…) | `http://<ip>:8080/v1/messages`, même clé |

Test depuis l'autre poste : `curl -H "Authorization: Bearer UNE-CLE…" http://<ip>:8080/v1/models` doit répondre un JSON ; `401` = clé fausse ou absente, « connection refused » ou délai dépassé = pare-feu, ou le serveur écoute encore sur `127.0.0.1`.
Alternative sans ouvrir le réseau : un tunnel SSH depuis l'autre poste, `ssh -L 8080:127.0.0.1:8080 utilisateur@serveur`, puis `http://127.0.0.1:8080/v1` (la clé reste conseillée).
Sans clé, tout poste qui atteint le port utilise le modèle : ne l'exposez pas sans `--api-key` (`--no-api-key` le permet, avec un avertissement).

## Quels types contient mon GGUF ?

Le moteur s'arrête avec un message « blk.N… est du type X, non géré » quand un tenseur a un type qu'il ne connaît pas. Pour voir tous les types du fichier (en-tête seulement, rien n'est chargé ; Python 3.8+, bibliothèque standard) :

```sh
python3 tools/gguf_types.py /data/le-modele.gguf
```

Envoyez cette sortie avec l'erreur : elle suffit à savoir quels types manquent.

## Si ça casse

| Message | Cause et remède |
| --- | --- |
| `strata-q36: blk.N.experts are Q4_K/Q4_K, which the native expert kernel do not take` | GGUF K-quant ordinaire (Q4_K_M…) : la projection « down » de certaines couches est en Q4_K, que le moteur d'avant le correctif refusait : `git pull` et recompiler (étape 3). Le noyau Q4_K « down » est compilé mais n'a jamais tourné sur une carte |
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
| `--kv-gpu 1: only 1 CUDA device visible` | une seule carte visible : mettre `"--gpu", "0,1"` dans `args` du fichier de configuration (pas la clé `"gpu"`, qui ajoute `--layer-split`) |
| `--kv-gpu N … this build has no kernels for that card` | le binaire n'a pas l'architecture de la 2e carte : `tools/build_rhel7.sh --arch "61;XX"` (`nvidia-smi --query-gpu=name,compute_cap --format=csv`) |
| `attention scores GEMM failed` (ou `values`) | échec de cuBLAS dans l'attention. Le moteur à jour ajoute le statut, la carte, la VRAM libre et l'erreur CUDA éventuelle d'un noyau précédent : envoyez cette ligne et les lignes `--kv-gpu` du journal. `ALLOC_FAILED` = plus de mémoire sur la carte (contexte plus court, `--kv f16`) ; avec `--kv-gpu`, un test cuBLAS est fait dès le chargement sur la 2e carte |
| `not enough VRAM on device N for the K / V cache` | le contexte ne tient pas sur la 2e carte : `--kv f16`, un `--max-context` plus petit ou moins de `--slots` |
