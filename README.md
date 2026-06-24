# GPU Networking for Networker Dummies

> A deep dive into how GPUs talk to each other, written by a networking person, for networking people.

If you have spent your career thinking in terms of routers, MPLS labels, routing tables, BGP and the occasional `tcpdump`, the world of "GPU networking" can feel like it was designed by aliens. People throw around words like *NVLink*, *NVSwitch*, *collective*, *all-reduce*, *RDMA*, *RoCE* and *rail-optimized fabric* as if they were obvious. They are not.

This document is two things at once, the same way I did the [k8s service & LB testing notes](https://github.com/robric/k8s-svc-and-lb-testing) were:
- a **personal cheat sheet** so I (and maybe you) can stop re-clauding/googling "how does this NVLink thing is network or a bus?" every three months.
- an **educational source** to explain how GPU interconnects actually work, starting from networking intuition you already have.


The golden rule for the whole document: **whenever something looks like magic, we map it back to a networking concept you already know** — a link, a switch, a fabric, a routing decision, a congestion problem. GPUs are just a new kind of endpoint. The wires are still wires.

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

**Training — build the model.** This is the heavy one. Thousands of GPUs run in **lockstep**, grinding through the dataset for days or weeks, and after every step they reconcile what they learned — the **gradient all-reduce** from §3.7: *every parameter, summed across every replica.* Its traffic signature:
- **East-west and internal** — GPU↔GPU collectives dominate; almost nothing leaves for a user. Pure **backend** (§1.4).
- **Synchronized and bursty** — everyone hits the network at the same instant, then waits for the slowest before the next step (§4.1). **Tail latency sets the pace of the whole job.**
- **Throughput-bound** — you care about sustained bandwidth, not single-request microseconds.
- Networker's analogy: a giant **MPI/HPC batch job**, or a fabric-wide **bulk sync** where every node must reach the barrier before anyone moves on.

**Inference — use the model to serve users.** The model is trained; now you run it forward to answer requests. This *is* the **inference / serving** network already sitting at the top of the §1.4 diagram — **north-south, user-facing, load-balanced**, the Ethernet/IP/TCP traffic you've run your whole career. But there's a backend twist: a frontier model is still too big for one GPU, so even *serving* it is spread across GPUs — so inference *also* produces east-west collective traffic, just **lighter and less tightly synchronized** than training.

Inference also splits into **two phases** with different appetites — worth knowing because they're increasingly run on *different* GPUs:
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

**Option A: wire them directly to each other (a full mesh).** Give every GPU a cable to every other GPU. For a handful of GPUs this even works — early DGX-1 (V100) did a variant of it. But you already know how that ends — we don't cable every PC to every other PC, and the same reasons (O(N²) cabling, no clean way to extend) apply here. 

The one GPU-specific twist worth keeping in mind: a GPU's **18 links are a fixed budget**, so a mesh has to split them across all N−1 peers — at 8 GPUs that's ~2.5 lumpy links to each, and every GPU you add shrinks the bandwidth to the others. So, exactly as in the network, you put a **switch** in the middle.

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

Two things change from NVL72:

- **A second switch tier** (the spine) ties the per-rack leaf switches together, so every GPU still reaches every other — now in **two hops** instead of one.
- **The media splits by distance:** still **copper inside each rack** (the spine backplane from above), but **optical between racks** — copper can't span 8 racks, so the inter-rack links move to optics. (The very copper-reach limit we just hit, now solved with light instead of refused.)

And here's the payoff for the whole narrative: **the Clos / leaf-spine fabric you know from IP networking shows up *first inside scale-up*.** NVL576 is a folded Clos — but it carries **memory load/stores over NVLink**, not packets. Scale-out (§4) will be *another* Clos, carrying **packets over Ethernet/InfiniBand**. Same topology shape; different fabric, different payload.

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

# TODO list tracking


- different types of GPU with different capabilities ? they may not all have nvlink for example ? connectX vs BF ? 
- structure in 4. is not consistent with 3 (section names )
- we should talk about  the split inference (P/D A/F), KV cache -> dynamo 
- how much non nvidia the doc should be ???
- UAlink ?
- ESUN ?

