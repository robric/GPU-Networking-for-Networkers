# GPU Networking for Networker Dummies

> A deep dive into how GPUs talk to each other, written by a networking person, for networking people.

If you have spent your career thinking in terms of routers, MPLS labels, routing tables, BGP and the occasional `tcpdump`, the world of "GPU networking" can feel like it was designed by aliens. People throw around words like *NVLink*, *NVSwitch*, *collective*, *all-reduce*, *RDMA*, *RoCE* and *rail-optimized fabric* as if they were obvious. They are not.

This document is two things at once, the same way I did the [k8s service & LB testing notes](https://github.com/robric/k8s-svc-and-lb-testing) were:
- a **personal cheat sheet** so I (and maybe you) can stop re-clauding/googling "how does this NVLink thing connects GPUs to each other ?" every six months.
- an **educational source** to explain how GPU interconnects actually work, starting from networking intuition you already have.


The golden rule for the whole document: **whenever something looks like magic, we map it back to a networking concept you already know** — a link, a switch, a fabric, a routing decision, a congestion problem. GPUs are just a new kind of endpoint. The wires are still wires.

> **One vendor as the worked example.** "GPU" here means the category, not a brand. We build the whole mental model on **NVIDIA** hardware — it is the most widely deployed and the most concrete to point at, and its vocabulary (NVLink, SM, CUDA, NCCL) is what you will hit first in the wild. But the *architecture* is universal: a die of throughput tiles, fast on-package memory, a high-bandwidth **scale-up** link to nearby peers, and an **RDMA NIC** out to the cluster. AMD and Intel assemble the same building blocks under different names. We learn it once on NVIDIA, then map the alternatives — **AMD and Intel** in their own chapter, and the **open standards (UALink, Ultra Ethernet)** that answer NVIDIA's proprietary stack in theirs — once the model is in place.

---

## 1. The landscape: the GPU and the networks around it

Before we can talk about how GPUs *network*, we need two things: the bare minimum about what a GPU *is* (1.1–1.3), then the wider map — the **several different networks** an AI data center actually runs (1.4), the **workloads** that drive their traffic (1.5), and the **layered stack** this whole document climbs (1.6). The GPU bits are the *just enough* version — no warps, no occupancy, no kernel tuning. If you already know what an SM, HBM and `cuda:0` are, jump to 1.4.

### 1.1 Why GPUs run the show (and what the CPU still does)

A CPU has a few very fast, very clever cores optimized for *latency* — get one task done as quickly as possible, with big caches and branch prediction. A GPU flips the trade: it has **thousands of simple cores** optimized for *throughput* — do the same arithmetic on huge batches of data in parallel. AI training is exactly that: enormous matrix multiplications, the same operation over millions of numbers. That's why GPUs, not CPUs, run the show.

Networking analogy: think **control plane vs data plane**, exactly like in a router. The **CPU is the control plane** — relatively few cores making complex, branchy decisions and orchestrating the work. The **GPU is the data plane** — a wide, massively-parallel engine (like a forwarding ASIC) that just hammers the same operation across an enormous volume of data. The CPU decides *what* to run; the GPU does the bulk arithmetic. Different tool, different job.

But here's the **subtle difference that makes GPU networking its own beast** — and it's the thread for everything that follows. In a router, the data-plane payload is a **packet**, and a packet fits *entirely inside a single ASIC* while it's processed. One packet, one chip, done. An AI data-plane payload is different: the **tensors** (the model's weights and activations) can be too big to fit in one GPU, so they're **sharded** — split across many GPUs, each holding only a slice. No single GPU sees the whole thing.

That one fact changes everything. Because the payload is spread across chips, the GPUs can't work in isolation — they must **constantly collaborate**: exchange slices, sum partial results, redistribute outputs, all *mid-computation*. Where a router ASIC forwards independent packets that never need to know about each other, GPUs run a tightly-coordinated team effort. **That collaboration *is* the traffic GPU networking has to carry** — and it's a far richer, more demanding pattern than "forward this packet." The rest of this document is, fundamentally, about how that GPU-to-GPU collaboration gets wired and orchestrated.

### 1.2 What a GPU looks like, and the words for its parts

Before the glossary, one picture. At the highest level a GPU is **a big grid of compute tiles (Streaming Multiprocessor - SMs -) wrapped in a ring of very fast memory (HBM)**, with link interfaces (PCIe, NVLink) at the edges to talk to the outside world:

```
   +================ GPU (one device / one die) =================+
   |                                                             |
   | HBM stacks:   [====] [====] [====] [====]   (feed the grid) |
   | +------------------------------------------------------+    |
   | | SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM   |    |
   | | SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM   |    |
   | | SM  SM  SM   ...  ~100-150 SMs total  ...  SM  SM    |    |
   | | SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM   |    |
   | +------------------------------------------------------+    |
   |         shared L2 cache   (die-wide cache layer)            +===> NVLink : peer GPUs (scale-up)
   | HBM stacks:   [====] [====] [====] [====]                   |
   +====+==============================+=========================+
        |                              |
        | PCIe                         | PCIe (GPUDirect RDMA)
        v                              v
     host CPU                       NIC (ConnectX / BlueField)
                                       |
                                       | 800G  InfiniBand / RoCE
                                       v
                                 scale-out fabric
```

<p align="center"><em>A GPU: a grid of SMs wrapped in HBM, links at the edges.</em></p>

The three edges leaving the box map exactly onto the two-interconnect story coming up:

- **NVLink → peer GPUs** — the sideways link. This is *scale-up* (§3).
- **PCIe → host CPU** — the general-purpose link for control and loading data.
- **PCIe → NIC** (ConnectX / BlueField) — the on-ramp to the *scale-out* network (§4).

That last edge, the NIC, matters more than it looks. With **GPUDirect RDMA** the NIC reads and writes GPU HBM *directly* over PCIe, without bouncing through CPU memory — so GPU-to-GPU traffic across nodes never touches the host's RAM. On the newest Grace-Blackwell boards this gets tighter still (**Data Direct** / DirectNIC): instead of GPUs and NICs hanging off **discrete PCIe switch chips** (the older layout, BlueField-3 NICs, ~200 Gb/s per GPU), the **ConnectX-8 SuperNIC itself acts as the PCIe switch**, sitting directly in the GPU's PCIe path (~400 Gb/s per GPU) — fewer hops, no detour up to the CPU. But it's still PCIe (Gen6) electrically — *not* a new non-PCIe link, and *not* NVLink — just a more integrated PCIe topology.

Zoom into **one SM** — this is where the actual math happens:

```
   +------------------- one SM (Streaming Multiprocessor) -------------------+
   | Warp schedulers               -> pick which threads run each cycle      |
   | CUDA cores  [][][][][][][][]  -> general FP32 / INT lanes               |
   | Tensor cores  [####][####]    -> matrix-multiply engines (AI workhorse) |
   | SFU  [><][><]                 -> exp, log, sqrt... (e.g. softmax)       |
   | LD/ST units                   -> move data between regs & memory        |
   | Registers + Shared mem / L1   -> tiny ultra-fast scratch by the ALUs    |
   | Other units                   -> specialized: TMA copies, FP64, TMEM... |
   +-------------------------------------------------------------------------+
```

<p align="center"><em>Inside one SM: schedulers, math units, and local scratch memory.</em></p>

The takeaway: **compute is the grid of SMs; memory bandwidth is the HBM ring feeding them; the network (PCIe/NVLink) hangs off the edge.** Everything in this document is about that last part — the edge — but it only makes sense once you see that the edge exists to keep the HBM, and through it the SMs, fed.

Now the terms. A few recur everywhere — here's the minimum to read a spec sheet:

- **SM (Streaming Multiprocessor)** — the GPU's basic compute building block. One GPU has many SMs (think ~100–150 on a modern data-center GPU), each packed with arithmetic units. Roughly "a core, but really a cluster of cores." You rarely tune these as a networker; just know "more SMs = more compute."
- **CUDA core / Tensor core** — the arithmetic units inside an SM. **Tensor cores** are the specialized matrix-multiply units that do the heavy lifting for AI. When NVIDIA quotes "FLOPS", this is where they come from.
- **HBM (High-Bandwidth Memory)** — the GPU's own RAM, stacked right next to the chip on the same package. This is the GPU's equivalent of a server's DRAM, but enormously faster: **~3–8 TB/s** of bandwidth on current parts. Capacity is smallish (tens to ~192 GB), which is *why* models must be split across many GPUs — and that splitting is what creates GPU-to-GPU traffic in the first place.
- **VRAM** — informal synonym for HBM capacity ("this GPU has 80 GB of VRAM").
- **CUDA** — NVIDIA's programming model/software stack for running code on the GPU. For our purposes it's mostly relevant as the thing that *names and addresses* GPUs.

> **Why a networker should care about HBM:** the whole reason GPU networking exists is that one GPU's HBM can't hold a big model. You split the model across GPUs, and now those GPUs must constantly exchange data at speeds that *approach HBM bandwidth*. The network is there to feed the HBM. Keep that framing — everything downstream is about not starving these memories.

### 1.3 Host vs device

A GPU is not a standalone computer. It lives inside a server, attached to a CPU:

```
   +------------------------------- Server (one node) -------------------------------+
   |                                                                                 |
   |        CPU0 (host)  <===== UPI / Infinity Fabric =====>  CPU1 (host)            |
   |          |                                                  |                   |
   |          |  PCIe                                      PCIe  |                   |
   |    +-----+-----+-----+                          +-----+-----+-----+             |
   |    |     |     |     |                          |     |     |     |             |
   | +-----++-----++-----++-----+                 +-----++-----++-----++-----+       |
   | |GPU0 ||GPU1 ||GPU2 ||GPU3 |                 |GPU4 ||GPU5 ||GPU6 ||GPU7 |       |
   | |HBM  ||HBM  ||HBM  ||HBM  |                 |HBM  ||HBM  ||HBM  ||HBM  |       |
   | +-----++-----++-----++-----+                 +-----++-----++-----++-----+       |
   |  cuda:0 cuda:1 cuda:2 cuda:3                  cuda:4 cuda:5 cuda:6 cuda:7       |
   |                                                                                 |
   |\_____ NUMA domain 0 _______/              \_______ NUMA domain 1 _______/       |
   +---------------------------------------------------------------------------------+
```

<p align="center"><em>One node: two CPU sockets, eight GPUs, two NUMA domains.</em></p>

- The **CPU is the "host"**; each **GPU is a "device."**
- **Real boxes are dual-socket.** A reference 8-GPU server (NVIDIA HGX/DGX-class) has **two CPU sockets**, with the GPUs partitioned across them — typically GPU0–3 under CPU0 and GPU4–7 under CPU1 (often through PCIe switches). This is **not failover redundancy** — if a CPU dies its GPUs don't migrate. It's there for **PCIe lanes** (8 GPUs + ~8 NICs + NVMe need more lanes than one socket has) and **NUMA balance**. Note that this already makes the node a **NUMA machine** *before* NVLink enters — crossing from a CPU0-GPU to a CPU1-GPU traverses the inter-socket link (UPI / Infinity Fabric).
- Software enumerates the GPUs in a node as `cuda:0`, `cuda:1`, `cuda:2`, … — just an index per device, like interface IDs (`eth0`, `eth1`) on a box. This is what people mean by "N distinct GPUs": even when the GPUs are fully wired together, you still address `cuda:0 … cuda:7` individually. (Hold onto this — it's why "8 GPUs acting as one" is an abstraction, not a hardware fact.)
- **PCIe** is the general-purpose bus connecting CPU and GPUs (and NICs). It's fine for loading data and control, but it is **far too slow** to be the path GPUs use to share memory with each other at HBM speeds. Hold that thought — it's the exact gap NVLink exists to fill, in §3.
- *Aside:* "superchip" designs (Grace-Hopper, GB200) change this CPU↔GPU relationship — the CPU attaches to the GPU over a fast **NVLink-C2C** link instead of PCIe, at a different ratio (GB200 = 1 Grace CPU for 2 Blackwell GPUs). More in §3.6.

### 1.4 The two network paths: host vs GPU

Zoom out from the chip to the data hall, network-engineer hat on. A GPU node sits on *many* networks — but they fall into **two categories**, and the cleanest way to tell them apart is **which processor owns the path**. This is just **host vs device** (§1.3) drawn as networks:

```
   +============================ FRONTEND ============================+
   |                                                                  |   
   |  host / CPU path  ·  conventional Ethernet / IP / TCP   (-> §6)  |
   |                                                                  |
   |    inference / serving      users -> model endpoints (N-S)       |
   |    tenant / VPC             multi-tenant isolation, overlays     |
   |    orchestration / control  k8s · Slurm · job scheduling         |
   |    management / OOB         BMC · provisioning · telemetry       |
   |                                                                  |   
   +================================ ^ ===============================+
                                     |
                        via the CPU  |  (sockets, TCP/IP)
                                     |
              +----------------------+----------------------+
              |                 one GPU node                |
              |        host CPU        |        GPUs        |
              |        (host)          |      (devices)     |
              +----------------------+----------------------+
                                     |
                        via the GPU  |  (RDMA, GPUDirect)
                                     |
   +================================ v ===============================+
   |                                                                  |   
   |  GPU / RDMA path  ·  kernel-bypass, no sockets        (-> §3,§4) |
   |                                                                  |
   |    compute fabric           GPU <-> GPU  (the collective traffic)|
   |       · scale-up            NVLink, in-rack                (§3)  |
   |       · scale-out           IB / RoCE RDMA, cluster-wide   (§4)  |
   |                                                                  |
   |    storage fabric           GPU <-> high-perf storage            |
   |       · GPUDirect Storage   NVMe / parallel-FS -> GPU HBM (RDMA) |
   |                                                                  |   
   +============================= BACKEND ============================+
```

<p align="center"><em>Two paths off a node: the CPU frontend, the GPU/RDMA backend.</em></p>

- **Frontend — the host / CPU / socket path** (top of the diagram). Not one network but several, all conventional **Ethernet / IP / TCP** you already run: **inference / serving** (user requests to model endpoints, north-south, load-balanced), **tenant / VPC** (multi-tenant isolation), **orchestration** (k8s / Slurm scheduling), and **management / OOB** (BMC, provisioning). From a GPU's point of view it's all "stuff the host does for me" — and all of it is **your existing skill set** (→ §6).
- **Backend — the GPU / RDMA / memory path** (bottom). Kernel-bypass, no sockets, and itself **two fabrics**: a **compute fabric** carrying GPU↔GPU collective traffic — **scale-up** (NVLink, in-rack — §3) and **scale-out** (IB / RoCE RDMA, cluster-wide — §4); plus a **storage fabric** for **GPU↔high-performance storage** via **GPUDirect Storage** (NVMe / parallel-FS DMA'd straight into HBM). This is the new, hard part — the rest of the document.

Two things refuse to sit neatly on one side — which is exactly the tell that the split is about **path, not function**:

- **Storage shows up on both.** Bulk data-loading through the host CPU is *frontend*; the high-performance GPUDirect-Storage path is *backend*. Same data — different processor carrying it.
- **OOB management** is technically its own *out-of-band* wire (to the BMCs), but it's a host/management concern, so it rides with the frontend.

The takeaway that frames the whole doc: **the frontend is your existing skill set** (Ethernet, IP, BGP, load-balancing — back in §6); **the backend is the genuinely new thing** — the GPU datapath, in two fabrics (§3 scale-up, §4 scale-out). Everything hard lives on the path that skips the CPU.

### 1.5 The two workloads: training vs inference

Everything so far — sharded tensors, GPUs collaborating mid-computation — has quietly assumed *one* kind of work. But a GPU data center runs **two very different jobs**, and they stress the network in opposite ways. You already have the perfect pair of analogies for them: **training is a batch / bulk-sync job; inference is a latency-sensitive request-response service.** Which one you're wiring for changes what the network has to be good at.

```
   TRAINING  — backend only, all GPUs in lockstep
      G === G === G === G
      ||  gradient all-reduce  ||      east-west, synchronized every
      G === G === G === G              step  ->  tail-latency bound

   INFERENCE — users in front, lighter backend behind
      users  ->  [ model endpoints ]   north-south, load-balanced (frontend)
                       |
      G === G === G === G              east-west but lighter: the model is
      (lighter collectives)            still sharded across GPUs (backend)
```

<p align="center"><em>Training: backend lockstep. Inference: user-facing front, lighter back.</em></p>

**Training — build the model.** This is the heavy one. Thousands of GPUs run in **lockstep**, grinding through the dataset for days or weeks, and after every step they reconcile what they learned — the **gradient all-reduce** from §3.7: *every parameter, summed across every replica.* Its traffic signature:
- **East-west and internal** — GPU↔GPU collectives dominate; almost nothing leaves for a user. Pure **backend** (§1.4).
- **Synchronized and bursty** — everyone hits the network at the same instant, then waits for the slowest before the next step (§4.1). **Tail latency sets the pace of the whole job.**
- **Throughput-bound** — you care about sustained bandwidth, not single-request microseconds.
- Networker's analogy: a giant **MPI/HPC batch job**, or a fabric-wide **bulk sync** where every node must reach the barrier before anyone moves on.

**Inference — use the model to serve users.** The model is trained; now you run it forward to answer requests. This *is* the **inference / serving** network already sitting at the top of the §1.4 diagram — **north-south, user-facing, load-balanced**, the Ethernet/IP/TCP traffic you've run your whole career. But there's a backend twist: a frontier model is still too big for one GPU, so even *serving* it is spread across GPUs — so inference *also* produces east-west collective traffic, just **lighter and less tightly synchronized** than training.

For the models driving all this — **autoregressive LLMs**, which generate their answer one token at a time — inference splits into **two phases** with different appetites, increasingly run on *different* GPUs. *(This split is generative-specific, not universal: a classifier, an embedding model, or a vision model just does a single forward pass — no token-by-token decode, no KV cache. It's the autoregressive LLM that makes serving its own beast.)*
- **Prefill** — read the whole prompt and build its context in one parallel pass. **Compute-heavy, bursty** (like ingesting a full request payload at once).
- **Decode** — emit the answer one token at a time, each token depending on the last. **Latency-bound, many tiny steps, memory-bandwidth-hungry** (like a long-lived chatty session dribbling out its reply).

| Property      | Training              | Inference                       |
|---------------|-----------------------|---------------------------------|
| Goal          | build the model       | serve the model                 |
| Direction     | east-west (GPU↔GPU)   | north-south + lighter east-west |
| Network plane | backend               | frontend + backend              |
| Pattern       | synchronized, bursty  | streaming, request-driven       |
| Bound by      | throughput            | latency (especially decode)     |
| Looks like…   | HPC batch / bulk sync | a web request-response tier     |

The takeaway for the rest of the doc: **§3 and §4 are mostly the *training* story** — the synchronized backend collectives that push the fabric hardest. Inference adds the familiar **frontend** dimension (→ §6) plus a lighter backend. But "lighter" is changing fast: modern serving is starting to **disaggregate** — running prefill and decode on separate GPU pools and shipping the intermediate state (the KV cache) between them over the backend fabric — which turns inference into its own demanding network problem. We flag it here and come back to it later; for now, just hold the split: **training stresses the backend; inference spans both planes.**

### 1.6 The one-slide summary

> A node = 1 CPU host + several GPU devices (`cuda:0…`). Each GPU is thousands of throughput cores (SMs / tensor cores) fronted by a small pool of very fast memory (HBM). Models are too big for one GPU's HBM, so they're split across GPUs — which forces those GPUs to exchange data fast. **That "exchange data fast" requirement is the entire reason GPU networking exists** — and it splits into two problems, which is exactly where §2 begins.

---

## 2. GPU Networking, the big picture: two fundamentally different problems

When we say "GPU networking", we are actually talking about **two different interconnects** solving **two different problems**. Almost every confusion in this space comes from mixing them up. So before anything else, we separate them cleanly.

### 2.1 Scale-up vs scale-out, in one sentence each

- **Scale-up** = bind a *small number* of GPUs (8, 72, …) into a single **tightly-coupled shared-memory domain** — one global address space, memory-speed sharing — so they can cooperate on one problem *as if* they were one giant GPU. Note the *as if*: software still sees N distinct GPUs (the `cuda:0…` from §1.3), each with its own memory and scheduler; the fabric just makes "pretending" cheap. This is **NVLink / NVSwitch** territory.
- **Scale-out** = connect a *large number* of those domains together into a *cluster* of thousands or tens of thousands of GPUs, using a packet-switched network. This is **InfiniBand / RoCE-over-Ethernet** territory, and it is the part that looks most like the networking you already know.

A useful mental model from the networking world:

> **Scale-up is the backplane of a chassis switch. Scale-out is the spine-leaf fabric that connects many chassis together.**

Inside a chassis, line cards talk over a backplane that is fast, short, lossless and dumb-simple to reason about. Between chassis, you build a Clos fabric with cables, optics, switches, congestion control and routing. GPUs have exactly the same two layers.

### 2.2 A picture to anchor everything

```
                        ███  SCALE-OUT  ███
            (cluster of many nodes, packet-switched network:
                     InfiniBand or RoCE/Ethernet)

         Node A                                   Node B
   +-------------------+                    +-------------------+
   |   ███ SCALE-UP ███|                    |   ███ SCALE-UP ███|
   |                   |                    |                   |
   | GPU0 ==== GPU1    |     RDMA over      | GPU0 ==== GPU1    |
   |  ||   X    ||     | <================> |  ||   X    ||     |
   | GPU2 ==== GPU3    |   IB / RoCE NICs   | GPU2 ==== GPU3    |
   |   (NVLink mesh    |                    |   (NVLink mesh    |
   |    via NVSwitch)  |                    |    via NVSwitch)  |
   +---------+---------+                    +---------+---------+
             |                                        |
            NIC(s)                                   NIC(s)
             |                                        |
             +-------------- Leaf switch -------------+
                                  |
                              Spine fabric
```

<p align="center"><em>Scale-up binds GPUs inside a node; scale-out links the nodes.</em></p>

- The `====` and `X` **inside** each node are **NVLink** — the scale-up fabric. Memory-semantic, microsecond-and-below, hundreds of GB/s to TB/s *per GPU*.
- The lines **between** nodes are the **scale-out** network — packet-switched, RDMA, built from NICs, leaf and spine switches, measured in hundreds of Gb/s *per port*.

Two interconnects, two units even (GB/s vs Gb/s — note the capital B vs little b, we'll come back to that). Keep them separate in your head and 80% of the confusion disappears.

### 2.3 Why two layers at all? (the networking intuition)

Because the two problems have opposite requirements:

| Property            | Scale-up (NVLink)              | Scale-out (IB / RoCE)             |
|---------------------|--------------------------------|-----------------------------------|
| Goal                | Bind N GPUs into one           | Connect many nodes into a cluster |
|                     | shared-memory domain           |                                   |
| Distance            | Inside a box / rack            | Across racks, rows, the data hall |
| Semantics           | Memory load/store (address)    | Messages / RDMA (packets)         |
| Bandwidth per GPU   | ~0.9–1.8 **TB/s**              | ~0.4 **Tb/s** (400G) per NIC      |
| Latency             | 10s–100s of **ns**             | ~1–3 **µs**                       |
| Scale               | 8s to ~72 GPUs                 | Thousands to 100k+ GPUs           |
| Looks like…         | NUMA / shared-memory           |                                   |
|                     | multiprocessor                 | A Clos data-center network        |

You *cannot* build a 100,000-GPU machine entirely out of scale-up — the physics (distance, power, radix) won't let you. And you *don't want* to run tightly-coupled memory traffic over a routed packet network if you can avoid it — it's too slow. So you use the fast, dumb, short fabric where you can (scale-up), and the smart, routed, long fabric where you must (scale-out).

This document tackles **scale-up first** (NVLink), then scale-out later.

---

## 3. Scale-up: the NVLink fabric

> Goal of this section: by the end you should be able to explain, to another networking person, what NVLink *is*, what problem it solves, and why it is **not** just "a faster PCIe" and **not** quite "an Ethernet for GPUs" either.

### 3.1 The problem NVLink solves

Back in §1.1 we established the uncomfortable fact that drives all of this: tensors are **sharded** across GPUs, so the GPUs must **collaborate mid-computation** — exchanging slices and summing partial results constantly, every layer, while the math is running. And §1.2 gave us the speed those exchanges happen at: the data lives in **HBM**, which moves at **3–8 TB/s**. The collaboration traffic wants to run at something close to *that*, because the moment GPU-to-GPU transfer is much slower than HBM, the Tensor cores sit idle waiting for data — and idle Tensor cores are the one thing a $40k GPU must never do.

So the requirement is brutally simple to state: **let one GPU read and write another GPU's HBM at a useful fraction of HBM speed, with very low latency.** That's it. The question is just what wire you do it over.

**Why PCIe ran out of road.** PCIe is the obvious candidate — it's already there (§1.3) — but it fails on two counts:

- **Bandwidth.** A PCIe Gen5 x16 link is about **64 GB/s per direction** (~128 GB/s if you add both directions - nvidia math -). HBM is **3,000–8,000 GB/s**. So PCIe is roughly **30–60× slower** than the memory it's trying to feed. Routing the collaboration traffic over PCIe is like giving each line card in a chassis a single 1G uplink and asking it to keep up with a 100G backplane — the GPUs would spend most of their time stalled.
- **Topology.** PCIe is a **tree rooted at the CPU** (the root complex). GPUs don't talk to each other as equals; they talk *up* toward the CPU and back *down*, sharing the root's bandwidth. Even with peer-to-peer (GPUDirect P2P), you're squeezing many GPUs through a hierarchy that was designed for a CPU to reach its peripherals — not for 8 GPUs to all blast each other at full tilt simultaneously. It's an oversubscribed access network, not a non-blocking fabric.

**The goal: turn the tree into a memory fabric.** What NVLink sets out to do is replace that slow, CPU-rooted tree, *between the GPUs*, with a **flat, dedicated, high-bandwidth fabric** where any GPU can reach any peer's HBM directly — and do it with **memory semantics** (plain `load`/`store` to an address), not packet send/receive. In other words: make the GPUs a real **NUMA shared-memory domain** (the framing from §2), where "remote" memory is merely a few times slower than local, instead of dozens of times slower.

Here's the whole motivation in one table — one GPU's view of its three options for reaching data:

| Where the data is (one GPU's view) | Bandwidth          | Speed vs local HBM |
|------------------------------------|--------------------|--------------------|
| **Local HBM** (its own memory)     | ~3,000–8,000 GB/s  | 1× (baseline)      |
| **Peer GPU over NVLink**           | ~900–1,800 GB/s    | ~¼ – ½             |
| **Peer GPU over PCIe Gen5**        | ~128 GB/s (aggr.)  | ~1/30 – 1/60       |

That middle row is the entire point of NVLink: it drags "another GPU's memory" from *60× slower than local* up to *2–4× slower than local* — close enough that treating the whole group as one big pool of memory actually works. (Note the units callback from §2.2: HBM and NVLink are quoted in **GB/s**, the scale-out network later will be in **Gb/s** — a factor-of-8 trap waiting for the unwary.)

> **One-line version:** PCIe is a slow tree to the CPU; NVLink is a fast mesh between GPUs. Scale-up is the art of making "remote HBM" almost as cheap as "local HBM."

The next subsections unpack *how*: NVLink as a physical link (§3.2), how links become a fabric via NVSwitch (§3.3), what "memory semantics" really buys you (§3.5), and the real systems and numbers (§3.6).

### 3.2 NVLink as a *link*: lanes, sublinks, and how to read a spec sheet

NVLink looks exotic until you realize it's built from the **exact same Lego bricks as every other serial interconnect you know** — Ethernet, PCIe, InfiniBand. It's SerDes lanes, bonded into ports, bonded into a fat pipe. Once you see the hierarchy, the spec sheets stop lying to you.

**The hierarchy** — read top (the 2-wire pairs) down to bottom (the per-GPU pipe). This drawing shows the **NVLink 3.0+** layout — 4 pairs per sub-link; 1.0/2.0 used 8:

```
    PAIRS   each = 2 wires, ONE direction  (showing 4; an RX set mirrors it)
        | |  | |  | |  | |              
        [p]  [p]  [p]  [p]                  [p]  [p]  [p]  [p]
          \    \   /   /                      \    \   /   /
           \____\_/___/                        \____\_/___/
                |   bundle 4 same-way pairs          |
                v                                    v
                SUB-LINKS  (4 pairs, one direction each)
        +-----------------+                  +-----------------+
        |   TX sub-link   |                  |   RX sub-link   |
        +-----------------+                  +-----------------+
                  \                             /
                   \________ TX + RX __________/
                                |   pair one TX with one RX
                                v
                      LINK  (full-duplex "port")
                  +----------------------------+
                  |     1 link  =  TX + RX     |
                  +----------------------------+
                                |              
                                v
                           per-GPU NVLINK
        +--------------------------------------------+
        |  [L0][L1][L2][L3] .......... [L17]   (x18)  |
        +--------------------------------------------+
            = 18 x  50 GB/s  =   900 GB/s   (H100,      NVLink 4)
            = 18 x 100 GB/s  =  1800 GB/s   (Blackwell, NVLink 5)

   (NVLink 3.0+ shown: 4 pairs per sub-link; 1.0/2.0 used 8.)
```

<p align="center"><em>Wire pairs bond into sub-links, into links, into one per-GPU pipe.</em></p>

Map this to networking and it's familiar territory:

- A **differential pair** is one **unidirectional** SerDes wire pair. (Heads-up: a "lane" in PCIe/Ethernet usually means a *full-duplex* pair-of-pairs — one TX, one RX. NVLink counts the one-way pairs, so keep the direction straight.)
- A **link** is a **full-duplex port**: one transmit sub-link + one receive sub-link — exactly like a 400G Ethernet port that's really 4×100G lanes each way.
- The **per-GPU number** is **18 ports bonded into one logical pipe** — think a **LAG / port-channel** of 18 links. The GPU stripes its peer traffic across all of them.

**How to read the spec sheet without getting fooled.** NVIDIA quotes NVLink bandwidth as what it calls **"bidirectional"** — every link, both directions (TX + RX), summed. Heads-up: that is *not* how a networker uses the word. To us "bidirectional / full-duplex" is a property of the link, and we quote bandwidth **per direction**; the TX+RX sum we'd call **aggregate** or **total**. NVIDIA's "bidirectional" = your "aggregate." This naming gap is the source of 90% of the confusion when comparing NVLink to a NIC:

- **"Bidirectional" (NVIDIA) = aggregate TX+RX, not per-direction.** "900 GB/s" on an H100 is the TX+RX sum — i.e. **450 GB/s per direction**. A NIC quoted as "400G" is already **per-direction** (400 Gbit/s each way). So normalize first: H100 NVLink is ~3,600 Gbit/s *per direction* vs a 400G NIC's 400 — about **9× per direction**, not the ~18× the raw headlines imply.
- **GB/s, not Gb/s.** NVLink is **bytes**, NICs are **bits** — a factor of 8 (the §2.2 trap). 900 GB/s = 7,200 Gb/s.
- **Per-link vs per-GPU.** A spec might say "50 GB/s per link" *or* "900 GB/s per GPU." Same chip — just multiply by the 18 links.

†*Pairs-per-sub-link is well-documented as 8 for 1.0/2.0 and 4 from 3.0 on; the exact lane signaling rate of the newest gens varies by source, so treat that detail as approximate. The per-link and per-GPU totals are the solid, NVIDIA-published numbers.*

**The generations, in one table** (per-link and per-GPU are both *aggregated* with tx+rx as per NVIDIA's convention):

| NVLink gen | Year | GPU / arch        | Pairs/sub-link† | Links/GPU | Per-link (aggr.) | Per-GPU (aggr.) |
|------------|------|-------------------|-----------------|-----------|------------------|-----------------|
| 1.0        | 2016 | P100 (Pascal)     | 8               | 4         | 40 GB/s          | 160 GB/s        |
| 2.0        | 2017 | V100 (Volta)      | 8               | 6         | 50 GB/s          | 300 GB/s        |
| 3.0        | 2020 | A100 (Ampere)     | 4               | 12        | 50 GB/s          | 600 GB/s        |
| 4.0        | 2022 | H100 (Hopper)     | 4               | 18        | 50 GB/s          | 900 GB/s        |
| 5.0        | 2024 | B200 (Blackwell)  | 4               | 18        | 100 GB/s         | 1,800 GB/s      |
| 6.0*       | 2026 | R100 (Rubin)      | 4               | 18*       | 200 GB/s*        | 3,600 GB/s      |

\**NVLink 6.0 / Rubin is announced, not yet broadly shipping (full production early 2026). The **3,600 GB/s per GPU** figure is published (72-GPU Vera Rubin NVL72 → ~260 TB/s rack); the link count and per-link split shown (18 × 200 GB/s) are inferred from that total and may change.*

Notice *how* the bandwidth grows: from 2.0 to 4.0 the per-link rate was flat at 50 GB/s and NVIDIA just **added more links** (6 → 12 → 18). With 5.0 they ran out of "more links" headroom and instead **doubled the per-link rate** (faster ~200G-class SerDes lanes), keeping 18 links but reaching 1.8 TB/s — and 6.0 (Rubin) doubles the per-link rate again to land at 3.6 TB/s. Same two knobs any network architect has: *more ports*, or *faster ports*.

One honest caveat before we move on: a single NVLink **link only reaches one neighbor**. 18 links on a GPU does **not** mean it can talk to 18 GPUs at full speed by magic — it means it has 18 ports' worth of bandwidth that must be *distributed* across whatever peers it needs to reach. How those 18 links get wired so that all 8 (or 72) GPUs can talk to each other at full bandwidth is a switching problem — and that's **NVSwitch**, §3.3.

### 3.3 From links to a fabric: a single NVSwitch (one node)

We left §3.2 with a cliffhanger: a GPU has **18 links, and each link reaches exactly one neighbor.** So how do you get *all* the GPUs in a box talking to *all* the others at full bandwidth? This is a topology question you have answered a hundred times in networking — and the answer is the same one networking reached decades ago.

**Option A: wire them directly to each other (a full mesh).** Give every GPU a cable to every other GPU. For a handful of GPUs this even works — early DGX-1 (V100) did a variant of it. But you already know how that ends — O(N²) cabling that won't extend, and a GPU's 18 links are a fixed budget to split across every peer. No need to dwell on it: you put a **switch** in the middle (Option B).

**Option B: connect every GPU to a switch (NVSwitch).** NVSwitch is precisely that — a **non-blocking crossbar switch for NVLink traffic**. Every GPU plugs its 18 links into the switch tier instead of into other GPUs. The crossbar then lets **any GPU reach any other GPU at full NVLink bandwidth, uniformly** — no lumpy per-peer math, no favoritism:

```
   8-GPU HGX H100 — each GPU spreads its 18 links across 4 NVSwitch chips
   (the 4 chips together act as ONE non-blocking crossbar)

      G0    G1    G2    G3    G4    G5    G6    G7      8 GPUs, 18 links each
       \     \     \    |     |    /     /     /
        \     \     \   |     |   /     /     /         each GPU's 18 links
         \     \     \  |     |  /     /     /          go to ALL 4 chips
   +------------+------------+------------+------------+ (see zoom below for
   | NVSwitch 0 | NVSwitch 1 | NVSwitch 2 | NVSwitch 3 |  the 5+5+4+4 split)
   +------------+------------+------------+------------+
        => any GPU <-> any GPU, full ~900 GB/s, uniform
```

<p align="center"><em>Each GPU spreads its 18 links across four NVSwitch chips: any-to-any.</em></p>

In a real **8-GPU HGX H100 node** this is **4 third-generation NVSwitch chips**, and each GPU spreads its 18 links across all four — **5 links to two of the switches, 4 to the other two** (5+5+4+4 = 18, since 18 won't divide evenly by 4):

```
   Zoom — how ONE GPU's 18 links land on the 4 chips:

                     +---------+
                     |  GPU 0  |    18 NVLinks
                     +---------+
                    /   /   \   \
                  5/  5/   4\   \4      <- links per chip
                  /   /       \   \
             +----+ +----+ +----+ +----+
             | S0 | | S1 | | S2 | | S3 |
             +----+ +----+ +----+ +----+
                5  +  5  +  4  +  4  = 18      (every GPU wires up
                                                the same way)
```

<p align="center"><em>One GPU's 18 links, split 5+5+4+4 across the four chips.</em></p>

The result is **full bisection bandwidth**: all 8 GPUs can be talking to all others simultaneously, each at the full per-GPU rate, with no internal bottleneck.

If that picture feels familiar, it should — a **non-blocking crossbar is just a switch fabric**, the same principle inside any switch: the ports never cable to each other, they all connect to the fabric, and it sprays traffic across its planes so every pair gets full bandwidth. NVSwitch is that fabric, built for GPUs. The *only* real difference is the payload: a network fabric moves **packet cells**, NVSwitch moves **memory reads and writes**.

One physical detail: in this 8-GPU node the NVSwitch chips are **soldered onto the same baseboard as the GPUs** and wired by board traces, not cables. The whole scale-up fabric lives *inside the box* — a **"pizza-box" switch**, where the fabric silicon shares the board with the ports. That changes at rack scale (§3.4).

### 3.4 Scaling past the box: NVL72, then NVL576

You've climbed this ladder before. In networking, when a **single pizza-box switch** runs out of ports you move to a **modular chassis**; when the chassis runs out, you build an **IP fabric** — a Clos of many switches. Scale-up climbs the *same* ladder: §3.3 was the pizza box (8 GPUs, fabric on the baseboard), and now we take the next two rungs — the chassis (NVL72, §3.4.1) and the two-tier Clos (NVL576, §3.4.2).

#### 3.4.1 NVL72: one rack, one switch tier

The on-board NVSwitch handles 8 GPUs. To go bigger, NVIDIA lifts the same switch chips *out* of the server and into dedicated **NVLink Switch trays** wired across a whole rack. That's what a **GB200 NVL72** is: 72 Blackwell GPUs (18 compute trays) + **9 NVLink Switch trays**, all stitched into **one single NVLink domain** where any of the 72 GPUs can load/store any other's HBM at full speed. *This* is where "72 GPUs act as one giant GPU" (the abstraction from §1.1) stops being marketing and becomes a wiring diagram — it's a non-blocking NVLink fabric, just scaled from 8 ports to 72.

Physically, this is where the switch **leaves the baseboard**. The pizza-box collapses into a true chassis: compute trays and **NVLink Switch trays** become separate units in the rack, joined by the **NVLink spine** — a copper-cable backplane down the back. Now the chassis-router comparison is no longer an analogy — it's the literal build: **compute trays = line cards, switch trays = fabric cards, the spine = the backplane. The rack *is* the chassis.**

```
   GB200 NVL72 = the 8-GPU NVSwitch fabric, scaled 4 -> 18 chips:

     G0    G1    G2    ......    G69   G70   G71      72 GPUs
      |     |     |      ...      |     |     |        each GPU: 18 NVLinks,
       \    |     |              |     |    /          one to each chip
   ==== NVLink spine: passive copper backplane (~5,000 cables) ====
       /    |     |              |     |    \
      |     |     |      ...      |     |     |
   +-----+-----+-----+-------+-----+-----+-----+
   |NVS0 |NVS1 |NVS2 |  ...  |NVS15|NVS16|NVS17|      18 NVSwitch chips
   +-----+-----+-----+-------+-----+-----+-----+       (across 9 switch trays)

   each GPU  -> 1 link to each of the 18 chips
   each chip -> 72 ports = 1 from each GPU   =>  1,296 links, non-blocking
```

<p align="center"><em>NVL72: one link from each GPU to each of 18 chips, over copper.</em></p>

Zooming to a single GPU makes the "1 per chip" concrete (the §3.3 zoom, now 18 chips instead of 4):

```
   One GPU (G0) and its 18 NVLinks — exactly one to EACH chip:

                                 +---------+
                                 |   G0    |   18 NVLinks
                                 +---------+
                        1 /  1 / 1 |   ...   | 1 \ 1 \
                         /    /    |         |    \   \
                  +-----+ +-----+ +-----+     +-----+ +-----+ +-----+
                  |NVS0 | |NVS1 | |NVS2 | ... |NVS15| |NVS16| |NVS17|
                  +-----+ +-----+ +-----+     +-----+ +-----+ +-----+
                     1       1       1   ...     1       1       1      = 18 (1 per chip)

   G0 has NO direct link to any other GPU. It reaches all 71 others
   THROUGH the chips: every chip connects to all 72 GPUs, so
        G0  ->  any chip  ->  any GPU      (always one switch hop)
```

<p align="center"><em>Every GPU reaches all 71 others through the chips — one hop.</em></p>

**So how does a GPU actually attach to those switch trays? Still NVLink** — the same protocol as on the baseboard, just carried over *cable* now instead of board traces. Each GPU's 18 NVLink ports run out the back of its compute tray into the **NVLink spine**: a **passive copper** cable backplane of ~5,000 coax cables. There's no hand-cabling — trays mate through **rear blind-mate connectors**, so sliding a tray into the rack makes it "bite" into the spine.

And notice how *tidy* the wiring is (bottom of the diagram) compared with the 8-GPU node's lumpy **5+5+4+4**: at NVL72 there are 18 NVLinks per GPU and 18 NVSwitch chips, so it divides perfectly — **exactly one link per chip**, every chip touching all 72 GPUs.

This also quietly explains the rack itself: the spine is **copper**, and copper only carries NVLink signaling **a meter or two**. That's *why* NVL72 is one dense, liquid-cooled rack rather than a row of them. But copper's reach isn't the end of scale-up — only the end of *single-tier, single-rack* NVLink. The next rung breaks through it with optics **and a second switch tier**.

#### 3.4.2 NVL576: eight racks, two switch tiers (a folded Clos)

**Rubin Ultra NVL576** (announced, ~2027) puts **576 GPUs across 8 racks into one NVLink domain**.\* You can't do that with one tier of switches — 576 GPUs won't fan into a single bank of chips, and copper won't cross 8 racks. So NVIDIA does the exact thing a network architect does when one switch runs out of ports: **add a second tier.** The NVLink fabric becomes a **two-tier, all-to-all topology — a folded Clos (leaf-spine)**:

```
   NVL576 — two-tier NVLink Clos (folded): 8 racks, 576 GPUs

   tier 2 (spine):           [   NVLink spine switches   ]
                            /      |      |      |        \      <- OPTICAL
   tier 1 (leaf):    [ rack 0 ]  [ rack 1 ]  ...  [ rack 7 ]       between racks
                       | | |       | | |            | | |
                      72 GPUs     72 GPUs    ...    72 GPUs      <- COPPER in-rack
                       \____________ 576 GPUs total ___________/

   any GPU <-> any of the other 575  =  2 switch hops (leaf -> spine -> leaf)
```

<p align="center"><em>NVL576: a two-tier NVLink Clos — copper in-rack, optics between racks.</em></p>

Two things change from NVL72:

- **A second switch tier** (the spine) ties the per-rack leaf switches together, so every GPU still reaches every other — now in **two hops** instead of one.
- **The media splits by distance:** still **copper inside each rack** (the spine backplane from above), but **optical between racks** — copper can't span 8 racks, so the inter-rack links move to optics. (The very copper-reach limit we just hit, now solved with light instead of refused.)

**The Clos / leaf-spine fabric you know from IP networking shows up *first inside scale-up*.** NVL576 is a folded Clos — but it carries **memory load/stores over NVLink**, not packets. Scale-out (§4) will be *another* Clos, carrying **packets over Ethernet/InfiniBand**. Same topology shape; different fabric, different payload.

**And the ladder keeps climbing.** The generation after Rubin pushes further still: **Feynman's NVL1152** (~2028) links **8 next-gen "Kyber" racks into one 1,152-GPU NVLink domain**, and to reach that far the NVLink switches move to **co-packaged optics (CPO)** — light fused right into the switch package. Notice the through-line: every rung needs a *longer, faster* interconnect just to keep "remote HBM ≈ local HBM" —

> board traces (pizza box) → **copper** spine (NVL72) → **copper in-rack + optics between racks** (NVL576) → **co-packaged optics** (NVL1152).

Even so, scale-up still has a ceiling. You can keep stacking NVLink tiers, but each one costs more optics, power, and latency, and the NVLink domain stays in the hundreds-to-low-thousands of GPUs. Past that you stop extending the *memory* fabric and cross into the packet-switched **scale-out** network — a different fabric with different rules, which is why it gets its own half of the document. The hand-off is §3.7; the scale-out fabric itself is §4.

\**NVL576 / Rubin Ultra (~2027) and NVL1152 / Feynman (~2028) are announced roadmap, not shipping — treat the specifics as preliminary. The durable point is the **media ladder**: copper → optics → co-packaged optics, as the NVLink domain grows.*

### 3.5 Memory semantics: load/store vs send/receive

Four sections on *how the wires are arranged*. Now the part that genuinely breaks networking intuition: **what travels over those wires, and how software asks for it.** NVLink doesn't move *messages* — it moves *memory accesses*, and that one difference is what makes a pile of GPUs feel like a single machine.

There are two fundamentally different ways for one chip to get at data sitting in another:

**1. Message passing (send / receive) — the model you already know.** Both sides are active. The sender packages data into a message, addresses it, hands it to the network; the receiver posts a receive and copies it out. This is sockets, MPI, and — at the wire — every packet you've ever `tcpdump`'d. The defining trait: **the data is an explicit message, and the receiver has to participate.**

**2. Memory semantics (load / store) — the NVLink model.** There is no "send." A GPU just executes a **`load` or `store` to a memory address** — and if that address happens to live in *another* GPU's HBM, the NVLink fabric quietly fetches or writes it. The remote GPU is **passive**: it runs no code, posts no receive; its memory is simply *there*, in a shared address space. To the program, reaching a peer's HBM looks like reaching its own — just a few times slower (§3.1).

```
   MESSAGE PASSING  (send/receive — the network)
     GPU A                                  GPU B
     pack -> address -> SEND        ===>    post RECV -> copy out
     both sides active; the unit on the wire is a *message / packet*

   MEMORY SEMANTICS  (load/store — NVLink)
     GPU A                                  GPU B
     st [addr in B's HBM] = val     --->    ( passive — runs no code )
     ld R1, [addr in B's HBM]       <---    ( passive — runs no code )
     one side acts; the unit is a *memory access* to a shared address
```

<p align="center"><em>Send/receive needs both sides; load/store needs only the issuer.</em></p>

**Why this is the whole ballgame for scale-up.** The §3.1 goal was to make N GPUs a NUMA shared-memory domain — *this* is the mechanism. Because every GPU's HBM sits in one **global address space**, a pointer on GPU 0 can point into GPU 7's memory, and dereferencing it Just Works over NVLink. You don't "send the tensor to GPU 7" — you write to where GPU 7 will read it. That's why "72 GPUs act as one giant GPU" is more than a slogan: at the lowest level they share an address space the way cores in one chip do.

**The spectrum (and where RDMA fits).** It isn't a clean binary — the network has spent 20 years inching *toward* memory semantics:

| Model                    | Primitive                   | Who acts                  | Granularity              |
|--------------------------|-----------------------------|---------------------------|--------------------------|
| **Sockets / TCP**        | `send` / `recv`             | both CPUs                 | a message                |
| **RDMA**<br>(scale-out)  | one-sided<br>`READ`/`WRITE` | NIC only<br>(remote idle) | work-request<br>(~KB)    |
| **NVLink**<br>(scale-up) | `load` / `store`            | issuing GPU<br>only       | 1 instruction<br>(bytes) |

RDMA is the interesting middle: it's **one-sided** like NVLink (the remote CPU doesn't participate), which is why people call it "memory-like." But RDMA still moves **packets**, posted as work-requests through queue pairs, at kilobyte granularity and microsecond latency. NVLink is the far end: **instruction-level memory access**, at near-HBM granularity and tens-of-nanoseconds latency. RDMA reaches *toward* memory semantics; NVLink *is* memory.

**One honest clarification for the literal-minded networker.** "But surely it's still packets on the wire?" Physically, yes — NVLink frames its transactions into flits and signals them over the SerDes lanes (§3.2). There's a wire format. The difference isn't "no packets ever," it's **the abstraction exposed to software**: NVLink presents *memory* (addresses, load/store), not *messages* (sockets, send/recv). You never write networking code to use it — you dereference a pointer and the hardware turns that into fabric traffic. That line, **memory vs messages**, is the real boundary between scale-up and scale-out — deeper than bandwidth or distance.

> **Keep this:** scale-up = **load/store into a shared address space** (memory semantics, NUMA). Scale-out = **send/receive or RDMA of messages** (packets). Same goal — move bytes between GPUs — opposite programming models.

### 3.6 Decoding the names: DGX/HGX/MGX, Oberon/Kyber, and the NVL## trap

We've met the technology (§3.2) and the systems (§3.4). What's left is the part that makes NVIDIA's slides unreadable to a newcomer: **the names.** None of them are hard once decoded — here's the cheat sheet.

**DGX vs HGX vs MGX — three layers, not three products.** They name *different layers of the same stack*, which is why people use all three about "the same" machine:

- **HGX** = the **GPU baseboard** — the 8-GPU + NVSwitch board (§3.3) that NVIDIA sells to server makers. The building block, not a whole system.
- **DGX** = NVIDIA's **complete, branded system** built around an HGX board — the finished server/rack you buy from NVIDIA (DGX H100, DGX GB200).
- **MGX** = a **modular rack reference design** OEMs build to, so a Supermicro/Dell rack and an NVIDIA rack are mechanically compatible. Think "the spec for the rack," not a box.

> One line: **HGX = the board, DGX = NVIDIA's whole system, MGX = the rack blueprint.**

**Oberon vs Kyber — the rack architectures.** These name the *physical rack generation* the trays and spine plug into. **Oberon** is today's NVL72 rack (Blackwell, Rubin). **Kyber** is the next one (Rubin Ultra onward), denser — it roughly doubles the per-rack NVLink domain and is what NVL576 / NVL1152 are built on. When someone says "Kyber rack," hear "the post-Blackwell rack that holds more GPUs per NVLink domain."

**The `NVL##` trap — dies vs packages.** `NVL` + a number = the size of the **NVLink (scale-up) domain**. The trap is *what the number counts.* NVIDIA briefly counted **GPU dies**, then reverted to **packages**:

- A modern "GPU" **package is 2 dies** (Blackwell, Rubin). So "NVL72" (72 packages) and the short-lived "NVL144" (144 dies) were **the same rack**, counted two ways.
- So when the number jumps, ask *dies or packages?* before assuming the domain doubled. **NVL72 = NVL144 = one 72-package rack.**

**The whole lineup, on one line each** (NVLink-gen numbers and bandwidth are back in §3.2; this is just the name map):

| GPU generation          | Rack arch | Flagship NVLink domain | ~Year   |
|-------------------------|-----------|------------------------|---------|
| Blackwell (GB200/GB300) | Oberon    | NVL72                  | 2024–25 |
| Rubin                   | Oberon    | NVL72 (was "NVL144")   | 2026    |
| Rubin Ultra             | Kyber     | NVL576                 | 2027    |
| Feynman                 | Kyber     | NVL1152                | 2028    |

With the names decoded, a sentence like *"the Kyber-based Rubin Ultra NVL576 MGX rack"* stops being noise and parses cleanly: **Rubin Ultra GPUs**, in the **Kyber** rack design, wired into a **576-GPU NVLink domain**, to the **MGX** modular spec.

### 3.7 Where scale-up ends and scale-out must begin

Everything in §3 has been **one bounded thing**: a single NVLink memory domain — 8 GPUs, 72, eventually a few thousand — where any GPU can `load`/`store` any other's HBM as if it were local. Beautiful, fast, and *finite.* This last section is about the wall it hits, and what you do when you reach it.

**Why scale-up can't just keep growing.** Every rung up the ladder (§3.4) gets more expensive on several axes at once:

- **Power & cooling.** One NVL72 rack is ~**120 kW**, liquid-cooled. You cannot put a 100,000-GPU NVLink domain in a building — the power density alone is impossible.
- **Distance & media.** Copper dies at a meter or two; past that it's optics, then co-packaged optics — each step costlier and more power-hungry, just to hold "remote HBM ≈ local HBM."
- **Switch radix & latency.** Every extra NVLink tier (§3.4.2) adds hops and latency, and NVSwitch radix is finite. The shared-memory illusion weakens the bigger you stretch it.
- **Cost.** NVLink switching + optics is *premium* interconnect. Making all 100k GPUs in a cluster members of one memory fabric would be financially absurd.

So there's a hard ceiling — today, hundreds to a few thousand GPUs per NVLink domain. A frontier training job needs **tens of thousands**. The gap is bridged by *not* extending the memory fabric, but switching to a different one.

**The real architecture: scale-up islands, stitched by a scale-out fabric.**

```
   The real cluster = scale-up ISLANDS stitched by a scale-out FABRIC

   +-----------------+   +-----------------+        +-----------------+
   |  NVL72 island   |   |  NVL72 island   |        |  NVL72 island   |
   | 72 GPUs,        |   | 72 GPUs,        |  ...   | 72 GPUs,        |
   | NVLink memory   |   | NVLink memory   |        | NVLink memory   |
   | fabric (ld/st)  |   | fabric (ld/st)  |        | fabric (ld/st)  |
   +--------+--------+   +--------+--------+        +--------+--------+
            | NICs                | NICs                     | NICs
            +---------------------+------------ ... ---------+
                                  |
                  scale-out fabric: IB / RoCE, packet-switched
                  RDMA over a leaf-spine Clos   ->   all of §4
```

<p align="center"><em>The real cluster: scale-up islands stitched by a scale-out fabric.</em></p>

Inside each island, GPUs share memory over NVLink (§3.1–3.6). Between islands, they fall back to **message passing over the NIC** — RDMA packets across an InfiniBand or RoCE Clos (§4). Two fabrics, layered: a fast *memory* fabric inside the island, a routed *packet* fabric between islands.

**The punchline — the boundary is also where you cut the workload.** This isn't just physics; it dictates *how a model is partitioned.* You match each kind of parallel traffic to the fabric that suits it:

- **Tensor parallelism / MoE all-to-all** — chatty, fine-grained, latency-critical (an exchange *every layer*). This **must** live **inside** the scale-up island, on NVLink.
- **Data & pipeline parallelism** — coarser and less frequent (gradients once per step, activations once per stage). These ride **across** the scale-out fabric, where higher latency is tolerable.

And *that* is the real reason NVIDIA keeps pushing the NVLink domain bigger (8 → 72 → 576): **a larger scale-up island lets more of the chatty, hard traffic stay on the fast memory fabric**, leaving the scale-out network to carry only the coarse stuff. Grow the island, relax the network.

So scale-up ends not at a number, but at a **role boundary**: it handles the tight, memory-speed collaboration; everything beyond that is handed to the packet network. That network — RDMA, RoCE vs InfiniBand, rails, congestion control, the parts that look most like the networking you already do — is **§4**, the other half of this document.

---

## 4. Scale-out: the GPU cluster network

> Goal: by the end you should see the scale-out fabric for what it is — **a data-center network you already know how to reason about** (Clos, packets, ECMP, congestion control) — but pushed to extremes that break the usual assumptions, and speaking **RDMA** instead of TCP.

This half is home turf. Where scale-up (§3) was an alien *memory* fabric, scale-out is a **packet-switched network**: NICs, leaf and spine switches, links, routing, congestion control. You have built these. The twist is *what* runs on it and *how hard* it gets pushed:

- the endpoints are **GPUs, not servers**, and they talk **RDMA**, not sockets;
- the traffic is a handful of **enormous, synchronized flows**, not millions of small independent ones;
- the fabric is often required to be **lossless**, which is *not* how you built your last data center;
- and a few giant flows wreck plain **ECMP**, so the fabric needs help — either a **rail-optimized** topology or a flatter Clos with **adaptive load balancing** (adaptive routing / packet spraying).

Same golden rule as §3: map each piece to networking you already know, then flag exactly where GPU clusters diverge — because the places they diverge are where all the pain (and all the interesting engineering) lives.

*Section outline:*

- 4.1 The job: connect the islands — scale, and what actually crosses the boundary.
- 4.2 RDMA on the wire: one-sided, kernel-bypass, and GPUDirect.
- 4.3 InfiniBand vs RoCE: two ways to carry RDMA at scale.
- 4.4 Why AI traffic breaks ordinary networks: elephant flows, incast, synchronized bursts.
- 4.5 Keeping it lossless: PFC, ECN / DCQCN, and where it's heading.
- 4.6 Topology & load balancing: why ECMP isn't enough, then the two answers — rail-optimized fabrics vs flat Clos with adaptive LB (adaptive routing / packet spraying).
- 4.7 Collectives on the wire: what NCCL actually sends (and in-network reduction).

### 4.1 The job: connect the islands

§3.7 left us with the picture: **scale-up islands** — each an NVLink memory domain of 8–72 GPUs — that now have to be wired into a **cluster**. That's the whole job of scale-out. The numbers are what make it hard.

**The scale.** A frontier training run wants **tens of thousands** of GPUs; the biggest clusters are now **100,000+**. At 72 GPUs per NVL72 island, 100k GPUs is roughly **1,400 islands** to interconnect. And each GPU brings its **own NIC** — well-provisioned AI clusters run about **one NIC per GPU** (the NIC edge from §1.2) — so the fabric is terminating on the order of **100,000 high-speed ports**, all for a *single job*. That's a bigger network than most enterprises run in total.

**The bandwidth cliff (why the boundary exists at all).** Per GPU, the two fabrics aren't remotely close:

| Fabric                          | Per-GPU bandwidth            |
|---------------------------------|------------------------------|
| Scale-up (NVLink 5, Blackwell)  | ~1,800 GB/s (≈ 14,400 Gb/s)  |
| Scale-out (one 800G NIC)        | ~800 Gb/s                    |

That's an **~18× drop** at the island edge (and it grows with each NVLink generation — Rubin's 3,600 GB/s makes it ~36×). Cross the boundary and your bandwidth falls by more than an order of magnitude. *This* is the quantitative reason for §3.7's workload-cut rule: keep the chatty traffic **inside** the island; push across the NIC only what you must.

**What actually crosses.** Mostly the coarse, periodic collective traffic from §3.7:

- **data-parallel gradient all-reduce** — once per training step, but it's *every* parameter, summed across *every* replica;
- **pipeline activations** — handed stage to stage;
- and, when a model's tensor/expert-parallel group is forced larger than one island, some of that too (avoided where possible — that's the cliff again).

**The thing that makes it brutal.** These are not independent flows. A collective is **synchronized**: thousands of GPUs launch the same all-reduce at the same instant, blast the network, then **all wait for the slowest one** before the next compute step can start. The fabric isn't carrying background traffic — it sits on the **critical path of every iteration**, and its **tail latency sets the pace of the entire job**. A single congested link doesn't slow one flow; it stalls *all* the GPUs waiting on that collective. Hold onto that — it's the reason §4.4–4.6 exist.

### 4.2 RDMA on the wire: one-sided, kernel-bypass, GPUDirect

Back in §3.5 we slotted RDMA onto the spectrum between sockets and NVLink: **one-sided** like NVLink (the remote stays passive), but still **moving packets** at kilobyte granularity and microsecond latency. That was *where it fits*. This is *what it actually is* — because RDMA is the language every endpoint on the scale-out fabric speaks, and three tricks make it work.

**Why sockets can't do this job.** Trace a byte over plain TCP: the app makes a syscall, the kernel's TCP/IP stack processes it, the data is copied (often twice), the CPU fields interrupts, and the same happens in reverse on the far side. At a few Gb/s on a web server, fine. At **400–800 Gb/s per NIC**, the CPU would burn *all* its cycles just shuffling bytes — the host becomes the bottleneck before the wire is half full. Scale-out throws that whole model out.

```
   TCP / sockets  (the slow path you know)
     app -> syscall -> kernel TCP/IP -> copy -> NIC  ===>  NIC -> kernel -> copy -> app
            the CPU touches every packet: syscalls, copies, interrupts

   RDMA + GPUDirect  (the scale-out path)
     GPU HBM ==DMA==> NIC  ================>  NIC ==DMA==> GPU HBM
            CPU posts the request once, then never touches the data
```

<p align="center"><em>TCP touches the CPU on every packet; RDMA+GPUDirect keeps it off.</em></p>

**One idea, three cuts.** What follows isn't three separate inventions — it's a single goal, **no CPU touches the bytes on either end**, enforced at the three places a CPU could otherwise sit on the datapath. A router already keeps its data plane off the route processor; scale-out pushes that further, off *every* CPU, near and far:

| Where a CPU could sit on the path                       | What removes it              | Side |
|---------------------------------------------------------|------------------------------|------|
| **local** CPU — its kernel/OS (syscalls, copies, TCP) | **kernel bypass**            | near |
| **local** CPU — its DRAM (a staging buffer)           | **GPUDirect RDMA**           | near |
| **remote** CPU — running receive-side code            | **one-sided `WRITE`/`READ`** | far  |

Two cuts on the near side (the kernel, then the memory), one on the far side. They're genuinely different mechanisms — bypass is about *software on the path*, GPUDirect about *where the buffer lives*, one-sided about *who runs code* — but they add up to one outcome: a GPU writing into another GPU's memory with no processor in the loop.

**Cut 1 — kernel bypass (near side: the local OS) — which is just how a router already works.** You invented this one. A router has *never* run its data plane through the CPU: packets are switched in the line-card ASIC at line rate, and only control-plane or exception traffic gets **punted** up to the route processor. Kernel bypass is that exact split brought to the server. The NIC becomes the line-card fast path — the application talks to it *directly* through memory-mapped queues, with **no syscall and no kernel on the data path** — and the host CPU/kernel is demoted to route-processor duty: it sets the connection up once (the control plane, §1.1), then never sees a packet again. After setup, posting a transfer is just dropping a descriptor into a queue the NIC is already watching — no per-packet OS involvement, no copies. Servers traditionally did the *opposite*, pushing every byte up through the kernel; RDMA makes the server NIC behave like the router you already run.

**Cut 2 — GPUDirect RDMA (near side: the local DRAM).** Bypass took the local CPU's *software* off the path; GPUDirect takes its *memory* off too. The NIC DMAs **straight into and out of GPU HBM** over PCIe (the NIC edge from §1.2), so a cross-node GPU-to-GPU transfer is **HBM → NIC → wire → NIC → HBM** — the host's DRAM is never a stop on the way. This is the literal mechanism behind §1.4's "backend = the path that skips the CPU": the CPU sets the transfer up and is then completely out of the datapath.

**Cut 3 — one-sided operations (far side: the remote CPU).** The near-side cuts freed the *initiator's* CPU; this one frees the *target's*. RDMA's headline verbs are **`WRITE`** and **`READ`**: the initiating NIC writes into, or reads out of, the *remote* node's memory **without the remote CPU doing anything** — it runs no code and fields no interrupt (the §3.5 "one-sided" property). There's also a two-sided **`SEND`/`RECV`** pair for when both ends should participate (handshakes, control):

| Operation                   | Who acts       | Remote CPU     | Typical use           |
|-----------------------------|----------------|----------------|-----------------------|
| `SEND` / `RECV` (two-sided) | both ends      | posts a `RECV` | control, handshakes   |
| `WRITE` (one-sided)         | initiator only | passive        | push data into a peer |
| `READ` (one-sided)          | initiator only | passive        | pull data from a peer |

Each verb rides the wire as an **opcode in the RDMA transport header — the BTH (Base Transport Header)** — alongside the destination queue-pair number and a packet sequence number. That byte layout, and how InfiniBand and RoCE wrap the *same* BTH in different lower layers, is exactly §4.3.

The shape under the hood is just that: **app ↔ NIC via memory-mapped queues (queue pairs), NIC ↔ NIC over the wire, the OS nowhere in sight.** The one prerequisite: target memory must be **registered** with the NIC first, yielding an `rkey` the initiator presents to touch it — a **capability token** (an address *plus* pre-authorized permission) that §4.3 will show riding in the packet itself.

Put the three cuts together and that's the scale-out endpoint: a GPU writing into another GPU's memory a row of racks away, **no processor in the loop** — the closest a packet network gets to NVLink's memory semantics. But it rests on one quiet assumption: those one-sided `WRITE`s only stay correct if packets **don't drop**, and the simple RDMA transport has no graceful loss recovery. That assumption drives the rest of §4 — starting with *how* RDMA is actually carried.

### 4.3 InfiniBand vs RoCE: two ways to carry RDMA

Two carriers do that job in production — and the most useful thing to know up front is that **they share the same transport**; the difference is only what you wrap around it.

- **InfiniBand (IB)** — a purpose-built, end-to-end fabric (NVIDIA/Mellanox): its own NICs (HCAs), its own switches, its own link and network layers, **lossless by design.**
- **RoCE** — **RDMA over Converged Ethernet**: the *same* RDMA transport repackaged to ride the **Ethernet/IP** you already run. "RoCE v2" is the version everyone means today.

**The packet tells the whole story.** Lay an IB frame next to a RoCEv2 frame and the relationship is obvious — RoCEv2 takes InfiniBand's transport (the BTH and everything above it) *untouched*, and swaps only the **carrier** beneath it:

```
   On the wire (left = first byte out):

                  carrier  (differs)       |  transport  (byte-identical)
   InfiniBand   [ LRH ][ GRH* ]            |  [ BTH ][ RETH ][ payload ][ ICRC ]
   RoCE v2      [ Eth ][ IP ][ UDP:4791 ]  |  [ BTH ][ RETH ][ payload ][ ICRC ]

   * GRH is optional (inter-subnet). IB adds a trailing link VCRC;
     RoCE rides inside an ordinary Ethernet frame (FCS).
```

<p align="center"><em>IB and RoCEv2 share the transport; only the carrier beneath differs.</em></p>

Read left-to-right, that's the derivation made literal: **RoCEv2 = InfiniBand's transport layer dropped onto UDP/IP/Ethernet.** Everything from the **BTH** rightward is the same transport — same headers, same field layout (with a few fields *used* differently, noted just below); IB carries it over its own link/network headers (LRH/GRH), RoCEv2 carries it inside a normal UDP datagram — **destination port 4791**, the registered "this payload is RDMA" marker.

**Zoom into the BTH itself** — 12 bytes, three 32-bit words. The *layout* is identical whether InfiniBand or a RoCEv2 UDP datagram carries it; a few fields are *interpreted* differently, which we flag right after:

```
    0                   1                   2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |    OpCode     |S|M|Pad| TVer  |             P_Key             |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |F|B| Reserved  |                Destination QP                 |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |A|  Reserved   |          Packet Sequence Number (PSN)         |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

   OpCode = which RDMA op (the verb)   P_Key = partition (tenant) tag
   F/B = FECN/BECN congestion bits     A     = AckReq
   S/M/Pad/TVer = small control bits
```

<p align="center"><em>The 12-byte BTH: opcode, destination queue pair, sequence number.</em></p>

The fields that matter to us *are* the §4.2 concepts, now as bytes on the wire:

- **OpCode** — *which verb* (the `WRITE`/`READ`/`SEND` from the §4.2 table); it also tells the receiver whether an extended header like RETH follows.
- **Destination QP** — *which queue pair* (which connection), the 24-bit address of the target's queue.
- **PSN** (packet sequence number) — *ordering and loss detection* — the field that makes "don't drop packets" enforceable (→ §4.5).
- the **RETH** on an RDMA op carries the **remote virtual address + R_Key** — the `rkey` capability token from §4.2, now a header field.

That last one is worth drawing too, because the **RETH (RDMA Extended Transport Header)** *is* §4.2's one-sided superpower as bytes — 16 bytes saying *where* to write and *by whose leave*:

```
    0                   1                   2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                    Virtual Address  [63:32]                   |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                    Virtual Address  [31:0]                    |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                             R_Key                             |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                           DMA Length                          |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

<p align="center"><em>The 16-byte RETH: remote address, R_Key, length — where, and by whose leave.</em></p>

**Virtual Address** = where in the peer's HBM to land the data · **R_Key** = the `rkey` capability that authorizes it · **DMA Length** = how many bytes. It rides only on the *first (or only)* packet of an RDMA `WRITE`/`READ` — the BTH OpCode flags its presence. This one header is what lets the initiator reach straight into a peer's memory with the target CPU asleep (§4.2's far-side cut).

**Same bytes, not always the same meaning.** A handful of fields carry over structurally but diverge in use between the two fabrics:

- **F / B (FECN / BECN)** — the congestion bits, and only the *forward* half changes. **InfiniBand** sets **FECN (F)** in a data packet's BTH at the congested switch, and the receiver echoes back **BECN (B)** (piggybacked on an ACK, or in a CN packet). **RoCEv2** moves the *forward* signal up to **IP-header ECN** — the switch marks IP, so the BTH **F bit goes unused** — but the *backward* half is **still the B bit**: the receiver bounces a dedicated **CNP** (BTH opcode `0x81`, **BECN=1**) to throttle the sender. So **F is IB-only; B lives on both** — RoCE just packages it as the CNP (the §4.5 story).
- **P_Key (partition key)** — a real, Subnet-Manager-administered partition (≈ a VLAN) on InfiniBand; RoCEv2 has no Subnet Manager, so it's still *validated* on ingress but not SM-managed — isolation comes from VLANs / PFC priorities instead.
- **ICRC** — same field, different coverage: RoCEv2 recomputes it with IP/UDP standing in for IB's LRH and the GRH blanked to `0xFF`, since those bytes change hop-to-hop.

The same correspondence, layer by layer — note the transport (bottom) is structurally identical, only the carrier (top) differs:

| Layer                | InfiniBand                            | RoCE v2                               |
|----------------------|---------------------------------------|---------------------------------------|
| Link / local         | LRH — LIDs, set by a Subnet Manager   | Ethernet — MACs                       |
| Network / global     | GRH *(optional, IPv6-style GIDs)*     | IP — routable                         |
| Transport shim       | —                                     | UDP :4791 *(src port = flow entropy)* |
| **Transport (RDMA)** | **BTH** — opcode, dest QP, PSN        | **BTH** — identical                   |
| **RDMA addressing**  | **RETH** — remote virt addr + R_Key   | **RETH** — identical                  |
| Integrity            | ICRC + link VCRC                      | ICRC + Ethernet FCS                   |

**So what actually differs?** Only the carrier — but that one swap drives every practical tradeoff:

- **Routability — RoCEv2 has IP, so it rides your Clos.** Because the transport sits inside UDP/IP, a RoCEv2 packet is a *normal routable packet*: ECMP, leaf-spine, all of it. Better still, the **UDP source port is free entropy** — the NIC varies it per flow so ECMP can spread RDMA across parallel paths (a §4.6 lever). InfiniBand instead routes on its own **LIDs**, handed out by a centralized **Subnet Manager** — a single controller programming the fabric, a very different control plane from distributed IP routing. *(Aside: the original **RoCE v1** put the BTH straight over Ethernet with no IP — L2-only, non-routable; effectively dead. v2 added IP/UDP precisely to make RDMA routable.)*
- **Who guarantees losslessness — the real fork.** RDMA's simple transport assumes packets don't drop (§4.2). InfiniBand delivers that **for free**: its link layer uses **credit-based flow control** — a sender transmits only when the receiver has advertised buffer space, so the fabric *cannot* overflow and drop. Ethernet has no such thing; it drops when congested. So RoCE has to be **made** lossless on top, with PFC and ECN/DCQCN bolted on. That's not a footnote — it's the central operational headache of running RoCE, and it gets its own section (§4.5).
- **Ecosystem.** IB is one vendor's vertically integrated fabric, lossless out of the box at a premium; RoCE runs on **standard Ethernet switches from anyone** plus RDMA-capable NICs — commodity economics and your existing skill set — but now *you* own the losslessness IB handed you for free. (Who those vendors actually are is the landscape below.)

> **InfiniBand = buy a fabric that's lossless by construction. RoCE = make your Ethernet behave like one.**

**Stepping back: which do you buy — and from whom?** The mechanism splits cleanly; the *market* splits along **silicon**, and two camps matter:

- **NVIDIA — vertically integrated.** It sells *both* answers: **InfiniBand** (Quantum) and its own AI-Ethernet, **Spectrum-X** (a proprietary RoCE with adaptive routing and built-in congestion control). Choosing IB *or* Ethernet can still mean staying all-NVIDIA — Spectrum-X is what runs xAI's Colossus.
- **Broadcom — the merchant backbone.** Its **Tomahawk** (scale-out, now 102.4 Tb/s) and **Jericho** (scale-across) ASICs sit under almost everyone who isn't buying Spectrum-X — *including the hyperscalers' own switches.*

That last point is the easy one to get wrong: AWS, Google and Meta are **not** a third silicon camp. They wrap **Broadcom** switch chips in their own white-box hardware, their own network OS (Meta's FBOSS, SONiC), and — the part they genuinely own — their own NICs and transport (AWS's **Nitro** running the **SRD** protocol). The only fully in-house *silicon* on that path is the NIC/DPU and the accelerator's scale-up link (Google's **TPU** interconnect + optical-circuit switches), **not** the Ethernet switch.

Two things to carry out of here:

- **The silicon war is NVIDIA vs Broadcom** — fought mostly *inside* the Ethernet column now that NVIDIA sells Ethernet too.
- **The market has tipped Ethernet-majority.** IB held ~80% of AI clusters in 2023; by 2025 Ethernet leads back-end deployments, pushed by cost, multi-vendor choice, and the **Ultra Ethernet Consortium** standardizing an AI-grade transport. InfiniBand keeps the lowest-latency, turnkey crown — a premium niche — but "serious training = InfiniBand" is the 2023 picture: several of the *highest*-end trainers (Google's TPU pods, Meta's RoCE clusters, Anthropic on AWS Trainium) run Ethernet or custom fabrics, not IB — and even one-time IB strongholds like Microsoft and CoreWeave now run Spectrum-X alongside it.

Both fabrics deliver the GPU the identical RDMA verbs; the choice is about what you operate beneath them. The rest of §4 mostly assumes the harder, more common case — **RDMA on Ethernet** — because that's where the interesting failures live: why AI traffic breaks ordinary Ethernet (§4.4), and how PFC/ECN claw back the losslessness InfiniBand never had to (§4.5).

### 4.4 Why AI traffic breaks ordinary networks

§4.1 warned that the fabric sits on the **critical path of every iteration** and its tail latency sets the job's pace. Here's *why* that's hard: AI traffic violates the one assumption every data-center network you've ever built quietly relies on.

**The assumption: many small, independent flows.** Enterprise and cloud fabrics work because of **statistical multiplexing**. Thousands of users, millions of short, *independent* flows — so the aggregate is smooth, peaks rarely line up, and you can safely **oversubscribe** (provision for the average, not the sum). The law of large numbers does your capacity planning for you. And a little loss is fine: TCP backs off, retransmits, nobody notices.

**AI breaks every clause of that.** A training step is **one** workload, not millions of independent ones. Its flows are **few** (one or a handful per GPU pair), **huge** (gigabytes per collective, elephants not mice), **synchronized** (they start on the same clock edge and all want peak bandwidth at once), and **correlated** (no averaging-out — it's all the same job stepping in lockstep). The very first casualty is oversubscription: AI backend fabrics are built **full-bisection (non-blocking)** because you can't lean on statistical multiplexing — but as we'll see, even a non-blocking fabric still suffers the three pathologies below, because they're about *where and when* traffic lands, not average capacity.

**1. Elephant flows — and why they wreck ECMP.** ECMP spreads traffic by **hashing each flow** to a path. With millions of flows that averages out beautifully. With *eight* giant flows across *sixteen* links, hashing is a dice roll — two elephants collide on one link (running it at 200%) while others sit idle. One hot link throttles the whole collective:

```
   FEW ELEPHANTS — per-flow ECMP hashing can't spread so few flows:

       flow A ==\                          link 1:  A + B   ==>  200%  HOT
       flow B ===>--[ hash 5-tuple ]-->    link 2:  C       ==>  ok
       flow C ==/                          link 3:  ------   idle
                                           link 4:  ------   idle
       a handful of giant flows over many links: collisions, hot AND cold
```

<p align="center"><em>Too few flows for ECMP: elephants collide on one link, others idle.</em></p>

The UDP source-port entropy from §4.3 helps the hash, but more entropy can't fix *too few flows*. The real fixes are topology and smarter spreading — **§4.6**.

**2. Incast — many senders, one receiver, one instant.** Here's the subtlety, and it's worth getting right: a well-built **all-reduce does *not* cause this.** NCCL runs it as a **ring** (each GPU sends to one neighbor and receives from one — never N→1) or a **double-binary tree** (fan-in of ~2), *deliberately* shaped so no GPU is an N-way sink. Incast comes from elsewhere:

- **all-to-all** — the collective behind **MoE expert parallelism**: it runs *within an expert-parallel (EP) group* — the subset of GPUs that hold the sharded expert FFNs, not the whole cluster. Inside that group every GPU's router scatters its tokens to whichever members hold the chosen (top-k) experts, all at once — so each member really is receiving from many peers simultaneously. Genuine incast, just **bounded to the EP-group size**. It's already 20–60% of MoE training time, and worse once the group spans nodes.
- **port / uplink convergence** — even perfectly balanced flows pile up when routing lands several on one switch egress port, or on an oversubscribed spine uplink (the §1 elephant problem, now hitting a buffer).

Either way, N synchronized senders hit one port — and a switch buffer is **shallow**, tens of MB shared across dozens of 400G ports, only **microseconds** of absorption. It fills before any feedback loop can react, and overflows:

```
   INCAST — many flows hit one egress port at the same instant
   (all-to-all / MoE dispatch, or routing piling flows onto one port)

       GPU1 --\
       GPU2 ---\          +=====================+
       GPU3 ----+-------> |   switch egress     | -----> GPU0  (one receiver)
       ....  ---/         |   buffer ~tens MB   |
       GPUn --/           +=====================+
                            fills in microseconds at 400G  -->  DROP
```

<p align="center"><em>Incast: many synchronized senders overflow one shallow egress buffer.</em></p>

Storage networks have fought TCP incast for years; AI incast is worse, because it's **synchronized by design** and the transport beneath it (RDMA) despises drops. Stopping the overflow is **§4.5**.

**3. Synchronized microbursts.** Arrivals aren't random (Poisson) — they're **simultaneous**. A collective launch is a wall of traffic that fills buffers in microseconds: not *sustained* oversubscription you can provision around, but *instantaneous* oversubscription far faster than any congestion signal can chase.

**Why this is fatal, not merely slow.** Congestion hurts two different ways — and a collective amplifies both:

- **A drop is a cliff, not a hiccup.** RDMA's reliable transport tracks order by **PSN** (§4.3). Lose one packet and the receiver sees a gap in the sequence; the classic recovery is **go-back-N** — the sender rewinds to the lost packet and **re-sends everything after it**, including packets that already arrived fine (the receiver discarded them as out-of-order). One dropped cell can throw away a whole window of in-flight data — a throughput *cliff*, not TCP's gentle selective-retransmit dip. (Newer NICs add selective retransmit, but go-back-N is the baseline, and it's why §4.5 fights so hard to *never* drop.)
- **Even with zero drops, latency alone stalls everyone — the *tail*.** A collective is a **barrier**: an all-reduce isn't done until the *slowest* GPU's data lands, and **no GPU starts the next compute step until it's done.** So the number that matters is **tail latency** — the slowest path, not the average. A congested link that merely *queues* packets (dropping nothing) still delays its flow, makes that GPU the straggler, and **idles every other GPU in the job** until it catches up. Drops create a straggler; so does plain queueing delay — and the barrier turns *either* into a whole-cluster stall.

So "fast on average" is meaningless here. The fabric inherits two jobs ordinary networks never had: **don't drop** (engineer losslessness — §4.5) and **keep the tail short** by spreading the few giant flows (congestion control — §4.5; topology and load balancing — §4.6). Both exist for one reason — a single slow flow, whether *dropped* or merely *delayed*, is paid for by tens of thousands of waiting GPUs.

**Does inference break the network too? Yes — differently.** Everything above is *training*. Inference serves many independent requests, but a served model is **also sharded** (tensor-parallel, plus all-to-all if it's MoE) and **also on RDMA**, so it inherits the same drop-cliff and tail stakes. Two twists change the shape:

- **Batched and disaggregated.** Requests are processed in **continuous batches** (each forward pass steps a batch of active sequences in lockstep), and modern serving is often **disaggregated** — prefill and decode run on *separate* GPU pools, so the **KV cache** built during prefill is shipped prefill-pool → decode-pool. That transfer is a bulk, point-to-point flow ECMP collides like any elephant — a traffic class training simply doesn't have.
- **Smaller radius, latency as the metric.** The synchronized part is scoped to one serving instance (a TP/EP group of ~8–64 GPUs, not the whole cluster), and across independent requests **statistical multiplexing partly returns**. But the tail doesn't go away — it just becomes **user-facing**: the per-iteration all-reduce sits on every token's path (inter-token latency), and KV-cache transfer sits on time-to-first-token.

So inference is **hard to transport too**, just heterogeneously — smaller-radius collectives plus bulk KV-cache flows, against tight latency SLOs. §5 develops the full train-vs-inference traffic catalog; the point for *this* section is that **both** workloads break the ordinary-network assumptions, for overlapping reasons.

### 4.5 Keeping it lossless: PFC, ECN, and DCQCN

§4.4 left us with a mandate ordinary networks rarely had: **don't drop** (a lost packet is a go-back-N cliff) *and* **don't even queue for long** (a slow link is a barrier-wide stall). This section is how the fabric delivers the first half — losslessness — without the cure becoming the disease.

The two transports get there very differently, and it's **RoCEv2** that has the real work to do — Ethernet has to *retrofit* losslessness it was never born with. So we spend this section mostly on the RoCE machinery (**PFC**, then **ECN/DCQCN**), and only at the end circle back to how **InfiniBand** gets the same two jobs for free from its architecture. Everything until that closeout is the Ethernet/RoCE story.

Start from what you already run. Ethernet has had **flow control** since forever — **802.3x PAUSE**: a receiver whose buffer is filling tells the upstream port to *stop sending* for a moment. That's backpressure, and it's exactly the instinct here. The catch is that classic PAUSE stops the *whole link* — storage, management, RDMA, all of it. AI fabrics need to pause **only the RDMA traffic class** while everything else keeps flowing, and they need a gentler loop that keeps queues short so the hard PAUSE rarely fires at all. So losslessness on Ethernet is **two control loops at two timescales**, both scoped to a priority class:

```
   TWO CONTROL LOOPS, TWO TIMESCALES, ONE PRIORITY CLASS

   end-to-end  (slow, ~RTT)   queue builds -> ECN mark -> CNP -> sender slows
   hop-by-hop  (fast, <RTT)   buffer fills  -> PFC PAUSE -> upstream link stops

   ECN/DCQCN is the primary loop (keep queues short); PFC is the backstop
   (a hard PAUSE that must almost never fire) — tune ECN to trip first.
```

<p align="center"><em>Two loops, two timescales: ECN/DCQCN trims first, PFC is the backstop.</em></p>

**PFC (Priority Flow Control, 802.1Qbb) — the hard backstop.** This is 802.3x PAUSE you already know, made **per-priority**: each of the 8 Ethernet priority classes gets its own PAUSE, so the switch can freeze the lossless RDMA class (say priority 3) while TCP and management keep moving. When an ingress buffer for that class crosses a threshold, the switch fires a PAUSE *upstream*, hop by hop, and the link goes quiet for that class — **zero drops by construction**. It's fast (sub-RTT, no end-to-end loop) and blunt, which is exactly why it's the *last* resort, not the *first*:

- **Head-of-line blocking.** PAUSE stops *everything* in the class on that link, including flows that weren't headed for the congested port — innocent bystanders stall.
- **Congestion spreading.** A full buffer pauses its upstream, whose buffer then fills and pauses *its* upstream — the backpressure walks **backward across the fabric**, turning one hot port into a tree of paused links (the "victim flow" problem).
- **PFC deadlock.** If paused links form a **cycle** (a buffer-dependency loop, which credit-free Ethernet doesn't prevent), the whole cycle waits on itself forever — a fabric-wide freeze. Avoiding it takes careful topology and watchdogs. This is the nightmare PFC config tries to never reach.

So PFC is the airbag: it must exist, but if it's deploying often, something upstream is already wrong. The job of the second loop is to keep the airbag from firing.

**ECN + DCQCN — the proactive loop.** Instead of waiting for a buffer to fill, mark *early*. A switch whose queue crosses a (lower) threshold sets **IP-ECN = CE** on passing packets — the **forward** congestion signal from §4.3, the one RoCEv2 moved out of the BTH and into the IP header. The receiver sees the CE mark and reflects it to the sender as a **CNP** — and we already drew that packet: a **BTH with opcode 0x81 and BECN=1**, the *backward* signal. The sender's NIC runs **DCQCN** (a DCTCP cousin, implemented in NIC hardware): on each CNP it **cuts its injection rate**, then probes back up when the marks stop.

```
   DCQCN — the proactive loop that closes §4.3's CNP

                +------------- Switch fabric -----------+
                |                                       |
                |                                       |
   +--------+  data  +--------+     +--------------+    |   +--------+
   | sender |------->| switch |---->|    switch    |------->| recvr  |
   |  NIC   |   |    +--------+     | (congested)  |    |   |  NIC   |
   +--------+   |                   | marks ECN=CE |    |   +--------+
       ^        |                   +--------------+    |       |
       |        +---------------------------------------+       |
       |                                                        |
       +-<------- CNP: BTH opcode 0x81, BECN=1 -----------------+

   sender cuts injection rate -> queue drains -> CNP stops -> rate climbs
```

<p align="center"><em>The switch marks early; the receiver returns a CNP; the sender trims its rate.</em></p>

**Where the mark actually lives.** That forward "CE" signal isn't a new field — it's **two bits of the IPv4 header you've ignored your whole career**: the low 2 bits of the **DS field**. The top 6 bits are **DSCP** (RFC 2474); the bottom 2 are **ECN** (RFC 3168, which updates 2474):

```
     0   1   2   3   4   5   6   7
   +---+---+---+---+---+---+---+---+
   |          DSCP         |  ECN  |
   +---+---+---+---+---+---+---+---+

   ECN codepoint (RFC 3168, the 2 low bits):
     00  Not-ECT  (sender opted out)
     01  ECT(1)  ┐ ECN-capable, not yet marked
     10  ECT(0)  ┘
     11  CE       congestion experienced  <- switch sets this
```

<p align="center"><em>ECN is the low 2 bits of the IPv4 DS field; <code>11</code> = CE, "congestion experienced."</em></p>

The RoCEv2 sender ships its data marked **ECT** (`01`/`10`); a congested switch flips that to **CE (`11`)** *instead of dropping*; the receiver turns the CE into the **CNP** we just drew. Here's a real CNP on the wire (Wireshark, display filter `ip.dsfield.ecn == 1`) — and it comes with two gotchas worth knowing before you go looking for one:

```
Eth + IP Header + UDP (port 4791) 
[...]
   InfiniBand
     Base Transport Header
       Opcode: 129                       <- 0x81 = CNP                                             
       Partition Key: 65535
       Reserved (8 bits): 64             <- 0x40 = 0b0100_0000: the BECN bit (no
                                            tidy "BECN" field — it hides in here)
       Destination Queue Pair: 0x0000d2
       Packet Sequence Number: 0
     Vendor Specific or Unknown Header Sequence
       Data: 0000…0000  e42dad81         <- 16 reserved bytes (no payload) + ICRC

```
<p align="center"><em>A real CNP: opcode 129 (<code>0x81</code>); BECN hides in "Reserved = 64" (<code>0x40</code>); body is 16 zero bytes + ICRC.</em></p>

The whole trick is **threshold ordering**: set the ECN marking threshold *below* the PFC PAUSE threshold, so DCQCN starts trimming rates while there's still buffer headroom — and the hard PAUSE only fires if the gentle loop couldn't keep up (a microburst faster than one RTT). ECN/DCQCN is the everyday throttle; PFC is the floor under it.

**The other fabric, the same two jobs.** Step back: this section really has **two** jobs, not one — **losslessness** (never drop) and **congestion control** (keep injection rates sane so queues stay short). On RoCE those are two separate mechanisms — **PFC** for the first, **ECN/DCQCN** for the second. InfiniBand does *both* too; it just builds them into the architecture instead of bolting them on:

- **Losslessness — credits, not PAUSE.** The IB link layer is **credit-based**: a sender may transmit *only* when the downstream has advertised **credits** — guaranteed buffer space for what's coming. No credits, no send; a buffer can't overflow because nobody is ever allowed to send into a full one. Losslessness by construction — no thresholds to trip, no PAUSE to propagate, no deadlock to design around. RoCE's PFC is the Ethernet *emulation* of this.
- **Congestion control — IB's FECN/BECN loop.** IB runs the *same shape* of loop as DCQCN, but using the **BTH bits from §4.3**: a congested switch sets **FECN** in the BTH; the destination HCA, seeing it, returns a **BECN** to the source; the source throttles by bumping an index (**CCTI**) into a **Congestion Control Table (CCT)**, whose entries are **inter-packet delays** — it literally spaces its packets further apart. When the BECNs stop, a timer walks the index back down and the rate climbs. The table and timers are set by the **congestion-control manager** (the subnet manager's remit), not configured switch-by-switch — the same *SM-owns-the-fabric* pattern from §4.3.

This is exactly why the **F bit was IB-only** back in §4.3: IB does forward-marking *inside* the BTH, so FECN lives there; RoCE moved forward-marking out to **IP-ECN** and repackaged the backward half as the **CNP**, leaving the BTH's F bit dead on Ethernet. Same loop, different envelope:

| job                    | RoCE / Ethernet                     | InfiniBand                     |
|------------------------|-------------------------------------|--------------------------------|
| **losslessness**       | PFC — per-priority PAUSE (retrofit) | credits — lossless by design   |
| **congestion control** | IP-ECN mark → CNP → DCQCN cuts rate | FECN → BECN → CCTI/CCT spacing |

**Where this leaves us.** Losslessness handles §4.4's *first* axis — the drop cliff — and DCQCN starts on the second by holding queues short. But a single end-to-end rate loop can't fix flows landing on the *wrong path* in the first place: the elephant-on-ECMP collision and the few-giant-flows problem are about **placement**, not rate. Keeping the **tail** short needs the fabric to spread those flows across paths — first **shaping** it (§4.6), then **steering** traffic across it (§4.7).

### 4.6 Shaping the fabric: the Clos, and how GPUs hang off it

§4.5 ended with the placement problem: two elephant flows can hash onto the same link, and rate control alone won't pull them apart. Fixing that takes two steps. First you **shape** the fabric — decide where the switches and links go, which fixes where the paths are. Then you **steer** traffic across those paths (§4.7). This section is about shape.

The scale-out fabric is a **leaf-spine Clos** (a fat-tree). The GPUs' NICs plug into the leaves; the spine connects the leaves together. It's the same topology you already build, usually sized closer to non-blocking than an enterprise fabric because AI traffic doesn't statistically multiplex (§4.4). The one AI-specific question is which leaf each GPU's NIC connects to — and that choice is driven by the **scale-up fabric**, which an ordinary data center doesn't have.

#### 4.6.1 The baseline: a node homed to its leaf

Start from the design you already build. A server's NICs land on its top-of-rack switch — the leaf — and the spine ties the leaves together; dual-home to two leaves for the usual redundancy if you want it. Two GPUs inside the same node never touch the leaf: that traffic stays on NVLink (§3). The leaves and spine exist only to carry traffic *between* nodes.

```
             +=================================================+
             |  spine 0      spine 1      spine 2      spine 3  |
             +=====^============^============^============^=====+
                   |            |            |            |
               [leaf 0]     [leaf 1]     [leaf 2]     [leaf 3]
                 ||||         ||||         ||||         ||||
                 NICs         NICs         NICs         NICs
                 ||||         ||||         ||||         ||||
              [ node 0 ]   [ node 1 ]   [ node 2 ]   [ node 3 ]
                8 GPUs       8 GPUs       8 GPUs       8 GPUs
```

<p align="center"><em>Classical homing: each node's NICs land on its own ToR leaf; only cross-node traffic crosses the spine.</em></p>

This is **server-centric** homing, and it's the honest fallback when you have no strong scale-up fabric to exploit. It works for any workload and needs no special cabling rule. Its one weakness is where it puts the heavy traffic. A collective has GPU *k* of one node trading with GPU *k* of every other node — the traffic is **rank-aligned** (§5) — and here those rank-*k* NICs sit on different leaves, so it always climbs the spine. Rail-optimized homing (§4.6.2) is the one change that fixes exactly that.

#### 4.6.2 Rail-only: dropping the spine

The baseline spends a whole spine on rank-aligned traffic. Re-homing the NICs makes that spine unnecessary. Take one NIC per rank and change where it lands: GPU *k* of every **scale-up island** (the NVLink domain from §3.7 — 8 GPUs on an HGX box, 72 in an NVL72 rack) connects to the *same* switch — call it **rail *k***. Now rank-*k*-to-rank-*k* traffic, the bulk of a collective, stays on one switch, one hop. A rail is nothing exotic: one leaf switch with a single NIC from every island plugged into it.

That covers same-rank traffic. The other case is **cross-rail** — a GPU on rail *i* needs a GPU on rail *j*. Here the scale-up fabric does the work the spine used to: hop over NVLink to the in-island GPU that already sits on rail *j*, then send normally on rail *j*. That NVLink detour is NCCL's **PXN** (PCIe x Nvlink see §3). Cross-rail traffic rides scale-up, not a second switching tier.

So both cases are covered without a spine — same-rail on the rail switch, cross-rail over NVLink — and you can build the whole fabric with one switch per rail and nothing above it. Each island's GPUs hang off an **NVSwitch** (§3), which is what makes the cross-rail hop possible:

```
                                              +===============+
            +-----island A GPU0 -PCIe- NIC0---|               |
            |     island B GPU0 -PCIe- NIC0---| Rail 0 switch |
            |     island C GPU0 -PCIe- NIC0---|               |
            |                                 +===============+
            |
 +--------+ |                                 +===============+
 |NVSwitch|-+-----island A GPU1 -PCIe- NIC1---|               |
 |island A| |     island B GPU1 -PCIe- NIC1---| Rail 1 switch |
 +--------+ |     island C GPU1 -PCIe- NIC1---|               |
            |                                 +===============+
            |
            |                                 +===============+
            +-----island A GPU2 -PCIe- NIC2---|               |
                  island B GPU2 -PCIe- NIC2---| Rail 2 switch |
                  island C GPU2 -PCIe- NIC2---|               |
                                              +===============+

  cross-rail: A's GPU0 --NVLink--> A's GPU2 --NIC2--> Rail 2 switch  (no spine)
  ...  rails 3-7 same;  NVSwitch shown for island A only
```

<p align="center"><em>Every scale-up island has an NVSwitch (one shown); it lets a cross-rail hop (GPU0→GPU2) ride NVLink, not the spine.</em></p>

This is **rail-only**, and it is deliberately **not a traditional Clos**: there is no spine and no any-to-any second tier — just a set of independent rail switches stitched together by the scale-up fabric. That's the trade. You've swapped a whole spine layer for NVLink hops, which means rail-only stands or falls on scale-up:

- **Strong scale-up** (large NVLink domains) → cross-rail rides NVLink and the spine is genuinely optional.
- **Weak or no scale-up** → cross-rail has nowhere to go, so you keep the baseline's spine.

That coupling is why rail-only is NVIDIA-forward; vendors with smaller scale-up domains lean toward keeping the spine — the §8 vendor chapter gets into the specifics.

This rail-only block has a name: NVIDIA calls it a **scalable unit (SU)** — a set of islands sharing one set of 8 rail leaves. How many GPUs fit in one SU is set by the generation:

| Design          | Scale-up island | GPUs in one SU |
|-----------------|-----------------|----------------|
| DGX B200        | 8-GPU node      | 32 nodes = 256 |
| DGX GB200 NVL72 | 72-GPU rack     | 8 racks = 576  |

<p align="center"><em>One SU's GPU count: 32 B200 nodes, or 8 NVL72 racks.</em></p>

That's one rail-only block — same-rail traffic one hop, cross-rail rail-local over PXN, no spine. To go bigger you join several SUs with a spine; the result is a **SuperPOD**, and that spine — plus how far it scales — is §4.6.3. (Reference designs: [B200](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html), [GB200 NVL72](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/dgx-superpod-architecture.html).)

#### 4.6.3 Scaling past one pod: a spine over the rails

> **[raw material — to rework next]** the with-spine rail diagram and the radix-wall prose below are stashed here for the §4.6.3 pass. Mark 6 (cluster-sizing example, with verified switch radix) lands here.

```
          spine: extends each rail across the cluster (same-rail traffic)
                    ^             ^             ^             ^
                [ rail 0 ]    [ rail 1 ]    [ rail 2 ]    [ rail 3 ]
                    |             |             |             |
   island A         G0============G1============G2============G3
                    |             |             |             |
   island B         G0============G1============G2============G3
                    |             |             |             |
   island C         G0============G1============G2============G3
   ===  NVLink within a scale-up island (any rank <-> any rank)
    |   each GPU's NIC up to its rail-leaf switch
```

<p align="center"><em>Rail-optimized: GPU k of every island homes to rail k; same-rank traffic stays on one leaf, and NVLink carries the rest.</em></p>

**Where the paths are — and the radix wall.** As long as everything that needs to talk hangs off **one switch** (one rail), there's no spine and exactly **one path** between any two of them — nothing to balance. But a switch is finite: it tops out at its **port count** (~128 endpoints on a current 800G switch — the sizing math from earlier). Past that wall you need **several leaf switches joined by a spine** — and only *now* is it a real two-tier Clos, with **one equal-cost path per spine** between any two leaves.

So the spine is what *manufactures* multipath. Below the radix wall: one switch, one path, no spine needed. Above it: leaves plus spine, many paths. And the instant there are many paths, elephants pile onto the same one unless you spread them deliberately — that's **steering**, §4.7.

### 4.7 Steering the traffic: picking among the paths

**Load balancing — from per-flow hashing to per-packet spraying.** Topology lays the paths; load balancing assigns packets to them. The baseline you run everywhere — **per-flow ECMP** — is the exact thing §4.4 showed breaking: hash the 5-tuple, pin the whole flow to one path, and with a handful of elephants two collide while links idle. The fixes climb a ladder of finer granularity:

| granularity      | unit moved | reorder risk        | where you see it |
|------------------|------------|---------------------|------------------|
| per-flow ECMP    | whole flow | none                | today's default  |
| flowlet          | a burst    | low (gaps absorb)   | adaptive routing |
| per-packet spray | one packet | high (NIC reorders) | Spectrum-X / UEC |

- **Flowlet switching** splits a flow at the natural gaps between bursts and rebalances per burst; if a gap is longer than the worst path-delay difference, reordered packets can't overtake, so it's a safe win — when the gaps exist.
- **Adaptive routing** lets the *switch* choose the output port by **real-time queue occupancy** instead of a fixed hash. InfiniBand has done this in-fabric for years (SM-computed routes plus adaptive port selection — the *subnet-manager-owns-the-fabric* pattern again). On Ethernet it's recent: **NVIDIA's Spectrum-X** does per-packet adaptive routing — the Spectrum-4 switch picks the least-loaded port, the SuperNIC puts the packets back in order — and the open **Ultra Ethernet (UEC)** standard reaches the same end a different way — per-packet **spraying** that the NIC reorders (deep-dives in §8/§9).
- **Per-packet spraying** is the limit case: every packet of one flow is balanced independently across *all* equal-cost paths, so a 400G elephant smears across the whole fabric and no link runs hot.

```
   sender NIC:   p1 p2 p3 p4 p5 p6   one elephant flow, sprayed
                 |  |  |  |  |  |    packet-by-packet
                 v  v  v  v  v  v
              [ fabric: equal-cost paths; each switch sprays each packet
                out its least-loaded port -- fine-grain adaptive routing ]
                 |  |  |  |  |  |
                 v  v  v  v  v  v
   recv NIC:     p3 p1 p5 p2 p6 p4   arrives OUT OF ORDER
                 place each packet by its PSN/offset; report "done"
                 only when all have landed  ->  no go-back-N, spray is safe
```

<p align="center"><em>Spray every packet across all paths for balance; the receiving NIC restores order.</em></p>

**The catch is the one §4.4 named: reordering.** Spray a flow across many paths and its packets arrive **out of order**, because the paths differ in delay — and to RDMA's reliable transport an out-of-order packet looks like a **PSN** gap, i.e. loss, which trips the **go-back-N cliff** (§4.4). So spraying is only safe if the **receiving NIC** can swallow the disorder: write each packet to its destination address by its PSN/offset as it lands, in any order, and signal completion only once the last one arrives. NVIDIA does this as **Direct Data Placement** on its Spectrum-X SuperNICs; the open **Ultra Ethernet** (UEC) transport is built on the same trick (its own chapter, later). The switch sprays; the NIC un-sprays; the §4.4 cliff never fires.

**Where this leaves us — the scale-out toolkit, complete.** §4.4 handed the fabric two jobs ordinary networks never had: don't drop, and keep the tail short. Three mechanisms now cover them:

- **don't drop** — PFC, the lossless backstop (§4.5).
- **don't queue** — ECN/DCQCN holds injection rates so queues stay shallow (§4.5).
- **don't collide** — rail-optimized topology + adaptive/sprayed load balancing keep the few giant flows off each other's links (§4.6).

All three guard the same thing — the **tail** — because one slow flow, whether *dropped*, *queued*, or *collided*, is paid for by every GPU waiting at the barrier. That completes the scale-out *transport* story: how bytes cross the cluster without falling off the drop cliff or stretching the tail. What we have **not** described is the layer that *generates* those bytes — the **collectives** themselves (ring, tree, all-reduce, all-to-all) and how each maps onto the fabric. That's §5.


# TODO list tracking


- different types of GPU with different capabilities ? they may not all have nvlink for example ? connectX vs BF ? 
- structure in 4. is not consistent with 3 (section names )
- we should talk about  the split inference (P/D A/F), KV cache -> dynamo 
- how much non nvidia the doc should be ???
- UAlink ?
- ESUN ?

