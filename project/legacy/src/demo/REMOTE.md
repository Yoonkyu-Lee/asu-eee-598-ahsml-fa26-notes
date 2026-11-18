# Joining the community from home, over Tailscale (CP25, CP26)

We can't meet in person, so each of us runs a box at home and the boxes reach each other over Tailscale.

- **Yoonkyu's mini PC is box 0. It stays on all the time.** It plays the always-on home node of our design and holds the community while the rest of us are away.
- **Your laptop box only needs to be on when you want to use it.** That can be one joint session of about an hour (we'll pick a time), and then a few minutes whenever you like.
- When your box comes on, it is the same member as before: same identity, same posts. It gets what it missed from the boxes that are on, usually box 0, and it hands them anything they are missing from your last session. Nothing polls in the background (CP26).

First set up the box as in [LAN.md](LAN.md), "Before we meet": Python, `pip install`, then `fetch_models.py`. If you cloned the repo earlier, run `git pull` instead.

## 1. Join the tailnet (once)

1. Yoonkyu sends you an invite to his tailnet. Accept it and sign in.
   - Please join as a **user** of the tailnet. Don't ask for a shared device: a shared device can't open connections back to us, and our boxes push posts to each other.
2. Install Tailscale from https://tailscale.com/download and sign in with the account you accepted the invite with.
3. Find your Tailscale address. It starts with `100.`. Either look in the Tailscale menu, or run:
   ```
   .venv/bin/python src/demo/lan.py --whoami
   ```
   It prints your addresses and marks the Tailscale one. Post it in our group chat.

## 2. Start your box

Everyone uses the same list, in the same order. Your position in the list is your box number, 0, 1 or 2. Use the order we agreed in the group chat. The room password goes by DM or phone, never in the repo or a channel.

```
.venv/bin/python src/demo/lan.py --hosts 100.a.b.c,100.d.e.f,100.g.h.i --me 1 --room-pass "the password from the DM"
```
(Replace the three addresses with the real list from the group chat. The first one is box 0.)

- Then open **http://127.0.0.1:8601/** on your own laptop.
- In a second terminal, add `--check` to the same command. A box that is running says "reachable". A box that is off says "NOT reachable", which is fine: its owner may simply be away.
- On macOS, the first start asks whether Python may accept incoming connections. Click **Allow**. On Windows, allow Python on the network type Windows shows for the Tailscale adapter.

## 3. Coming and going

| What you do | What happens |
|---|---|
| Close the terminal (or Ctrl+C), start it again later with the same command | Same identity, same posts. You can still delete your own old posts. Your remaining slots carry over. Right after it starts, the box gets what it missed (usually under a second). Closing the lid works too: when the laptop wakes, the box notices and catches up. |
| Post, vote, search, room R | Same as in LAN.md. Searching never leaves your laptop. |
| Delete your post | It disappears from every box that is on. A box that was off deletes it when it catches up. The deleted post is removed from disk too, and no box hands it out again. |
| "Break the limit" | Your identity is revealed and dropped as before. It stays revealed after a restart. To get a new identity, start once with `--new-identity`. Your posts stay; you just can't delete the ones made under the old identity. |

Your box keeps its state in `project/state/box<your number>/`. This folder is not in git. It holds:
- `identity.json`: your identity secret. Never send this file to anyone.
- `keys.json`: one key per post you wrote, used to delete your posts. Never send this file either.
- `msgs.jsonl` and `img/`: the encrypted messages and photos your box received.
- `events.jsonl`: the measurement log, described below.

## 4. What we measure, and what to send at the end

After your last session, send Yoonkyu your **`state/box<i>/events.jsonl`** and nothing else. It holds times, message ids, message kinds, token counts, whether a post had a photo, and how long your CPU took to embed each post. It holds **no post text and no photos**. From the three logs, `src/demo/remote_report.py` works out:

- **How fast a new post becomes searchable on the other boxes that are on.** The target is 5 s for 99% of posts. Laptop clocks differ, so a box measures the difference with each peer when it comes on and corrects for it.
- **Round-trip time** between each pair of us, and how long a photo takes to arrive.
- **How long a box that was off takes to catch up** after it starts again.
- **How long your CPU takes** to embed a post with MiniLM and a photo with CLIP (Mac vs Windows).
- **How much we wrote:** counts, post length and the share of posts with photos. These are counts only, not content.

Please also send three things once, by hand:
1. Your laptop model and chip, e.g. "MacBook Air M2, 16 GB". Say whether it was usually plugged in.
2. The output of `tailscale ping <each other address>`. It shows whether our laptops talk **directly** ("via <ip>:<port>") or through a Tailscale **relay** ("via DERP").
3. Anything that broke or felt slow.

## What Tailscale is, and isn't, in this project

Tailscale only connects our laptops. The traffic between boxes is end-to-end encrypted (WireGuard), and Tailscale's servers can't read it. They do know which devices are in the tailnet. That makes Tailscale a stand-in for something our design has to do itself: in the real design, boxes find each other and get through home routers on their own, with no company in the middle. That is a later checkpoint. Until then, Tailscale is test scaffolding, not part of the design.
