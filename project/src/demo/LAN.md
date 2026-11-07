# Joining the community from your own laptop (CP24)

> Can't meet in person? See [REMOTE.md](REMOTE.md) to join from home over Tailscale (CP25). Set up as below first.

Each of us runs **one home node box** on our own laptop. The three laptops talk to each other directly. There is no server and no operator. Each box keeps its own copy of the posts, its own search index (MiniLM text embeddings and CLIP photo embeddings), and its own identity secret. Design and results: [cp/CP24.md](../../cp/CP24.md) (in Korean).

## Before we meet: set up once, about 10 minutes plus a ~720 MB download

You need Python 3.10 or newer. On a Mac: `brew install python@3.12`, or use the installer from python.org.

```
git clone https://github.com/Yoonkyu-Lee/asu-eee-598-ahsml-fa26-notes.git
cd asu-eee-598-ahsml-fa26-notes/project
python3 -m venv .venv
.venv/bin/pip install -r src/demo/requirements.txt
.venv/bin/python src/demo/fetch_models.py        # MiniLM + CLIP ONNX from Hugging Face, SHA-256 checked
```
On Windows, use `.venv\Scripts\python.exe` wherever this page says `.venv/bin/python`.

Check that the models are in place: `.venv/bin/python src/demo/fetch_models.py --check` should end with "all models in place".

## At the meeting

1. **Join the same network.** Campus Wi-Fi usually blocks traffic between laptops, so use one phone's hotspot. Both iPhone and Android hotspots work.
2. **Find your address.** Run `.venv/bin/python src/demo/lan.py --whoami`. On an iPhone hotspot the address looks like `172.20.10.x`.
3. **Agree on an order** for the list, for example `A,B,C`. Your position in the list is your box number: 0, 1 or 2.
4. **Start your box.** Everyone runs the same command; only `--me` changes. Put your own number after `--me`:
   ```
   .venv/bin/python src/demo/lan.py --hosts 172.20.10.2,172.20.10.3,172.20.10.4 --me 0 --room-pass "the password we said out loud"
   ```
   - Leave out `--room-pass` on one laptop. That box will still pass room R's encrypted messages along, but it cannot read them.
   - The first time you run it, the firewall asks about incoming connections for Python. Click **Allow**: on Windows choose Private networks, on macOS choose "Allow incoming connections". If you pick Deny, nobody can reach your box.
5. **Check you can reach the others.** In a second terminal, run the same command with `--check` added. Every other box should say "reachable".
6. Open **http://127.0.0.1:8601/** in a browser on your own laptop. Only your own laptop can use your box's screen. Requests that come in from the network are refused.

## Things to try

| Try | What you should see |
|---|---|
| Post (sub A), or attach a photo from your disk | The post shows up on the other two screens within a few seconds (screens refresh every 5 s). The photo is cropped to 224×224 and indexed with CLIP on every box. |
| ▲ / ▼ / delete | Votes add up on every box. Only the author can delete; a delete removes the post from every box. |
| Search → Text / Photos by words | Your box searches its own index. Nothing leaves your laptop. |
| Room R | Only boxes started with the password can read it. |
| Write a lot | New identities get 18 text slots and 2 photos per hour. A short post uses 1 slot; a post of 82 or more tokens uses 2. When the slots run out, the client tells you to wait. |
| **Break the limit** (bottom right) | Your box sends two posts on the same slot. The other boxes compute your identity secret from the two posts, publish it, and drop you. Your screen shows a red banner. Start your box once with `--new-identity` to get a new identity (CP25). |
| Stop a box, keep posting, then restart it | The restarted box catches up from the others within about 20 s. Since CP25 it keeps its identity, posts and keys in `state/box<i>/`, so it can still delete its old posts. |

## What is real and what is not

- **Real:**
  - Ed25519 signatures.
  - ChaCha20-Poly1305 encryption, with the room key derived from the password.
  - RLN shares and nullifiers in the BN254 field, so reusing a slot really reveals the secret.
  - MiniLM and CLIP inference on each laptop's CPU.
  - Gossip between the laptops over HTTP.
- **Not real:**
  - The zero-knowledge proof itself. Each message carries a "proof ok" flag instead of a Groth16 proof. The flag is false once your identity is removed from the group. Real proofs are planned for a later checkpoint.
  - The "chip time" on the work panel. It comes from our chip model; the laptop is not the chip.
- Your identity secret is created the first time your box starts and is never sent anywhere. Since CP25 it is kept in `state/box<i>/identity.json` on your laptop (not in git).

## If something does not work

- `--check` says "NOT reachable":
  - Are you on the same hotspot?
  - Did you allow Python through the firewall? On Windows, the hotspot network may be set to "Public", so allow Python on public networks too, or switch the network to Private.
  - Is that person's box running?
  - Is everyone's `--hosts` list exactly the same, in the same order?
- The box does not start: look at the error. "models missing" means you need to run `fetch_models.py`.
- On a Mac, if the INT8 MiniLM model fails to load, the box falls back to FP32 by itself. The log says so.
