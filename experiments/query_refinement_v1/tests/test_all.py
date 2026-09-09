import copy
import json
import math
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from ..core import ARMS, Spec, Update, attend, refined_read, rotate
from ..data import graph_rows, paired_summary
from ..pipeline import digest, qualify, require_receipt, train_arm, encode_rows
from ..runtime import Runtime
from .fixture import Fixture

torch.set_num_threads(1)


def tensors(dtype=torch.float32):
    torch.manual_seed(8)
    q, k, v = torch.randn(2, 4, 5, 8, dtype=dtype), torch.randn(2, 2, 5, 8, dtype=dtype), torch.randn(2, 2, 5, 8, dtype=dtype)
    pos = torch.arange(5)
    return q, k, v, pos, pos


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_end_alignment(backend, dtype):
    q, k, v, qp, kp = tensors(dtype)
    full = attend(q, k, v, qp, kp, backend=backend)
    tail = attend(q[:, :, 3:], k, v, qp[3:], kp, backend=backend)
    torch.testing.assert_close(tail, full[:, :, 3:])


@pytest.mark.parametrize("arm", ARMS)
def test_identity_and_immutability(arm):
    q, k, v, qp, kp = tensors()
    saved = [x.clone() for x in (q, k, v)]
    up = Update(8, 4, 3)
    cos, sin = torch.ones(1, 1, 5, 8), torch.zeros(1, 1, 5, 8)
    base = attend(q, k, v, qp, kp)
    out = refined_read(q, k, v, qp, kp, cos, sin, Spec(arm, 1, 3), up)
    assert torch.equal(out, base)
    for before, after in zip(saved, (q, k, v)):
        assert torch.equal(before, after)


@pytest.mark.parametrize("arm", ARMS[1:])
def test_feedback_gradients_and_counts(arm):
    q, k, v, qp, kp = tensors()
    up = Update(8, 4, 3)
    assert sum(p.numel() for p in up.parameters()) == 52
    optimizer = torch.optim.SGD(up.parameters(), lr=0.1)
    for step in range(3):
        optimizer.zero_grad()
        out = refined_read(q, k, v, qp, kp, torch.ones(1,1,5,8), torch.zeros(1,1,5,8), Spec(arm, 1, 3), up)
        out.square().mean().backward()
        assert float(up.up.weight.grad.abs().sum()) > 0
        if step > 0:
            assert all(float(p.grad.abs().sum()) > 0 for p in up.parameters())
        optimizer.step()
    assert bool(torch.isfinite(out).all())


def test_bounded_update():
    up = Update(8, 4, 3)
    with torch.no_grad():
        up.up.weight.fill_(100)
        up.gain.fill_(100)
    assert float(up(torch.randn(2,4,5,8)*1e6).detach().abs().max()) <= 1


def test_unconnected_old_queries_no_effect():
    q, k, v, qp, kp = tensors()
    first = attend(q, k, v, qp, kp)
    q[:, :, :-1] += 99
    second = attend(q, k, v, qp, kp)
    assert torch.equal(first[:, :, -1], second[:, :, -1])


def test_broadcast_key_and_value_identities():
    q, k, v, qp, kp = tensors(torch.float64)
    base = attend(q, k, v, qp, kp)
    torch.testing.assert_close(attend(q, k+0.37, v, qp, kp), base)
    torch.testing.assert_close(attend(q, k, v+0.37, qp, kp), base+0.37)


def test_repeated_queries_collapse_with_same_visibility():
    q, k, v, qp, kp = tensors(torch.float64)
    q = q[:, :, :1].expand(-1,-1,5,-1)
    out = attend(q, k, v, qp+5, kp)
    torch.testing.assert_close(out, out[:, :, :1].expand_as(out))


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_no_future_information(backend):
    q, k, v, qp, kp = tensors()
    before = attend(q, k, v, qp, kp, backend=backend)
    k[:, :, 3:] += 100
    v[:, :, 3:] -= 100
    torch.testing.assert_close(attend(q,k,v,qp,kp,backend=backend)[:, :, :3], before[:, :, :3])


@pytest.mark.parametrize("failure", ["duplicate", "negative", "float", "empty_visibility", "gqa", "backend", "scale"])
def test_fail_closed_attention(failure):
    q,k,v,qp,kp = tensors()
    kwargs = {}
    if failure == "duplicate": qp = torch.zeros(5,dtype=torch.long)
    if failure == "negative": qp = qp-1
    if failure == "float": qp = qp.float()
    if failure == "empty_visibility": kp = kp+10
    if failure == "gqa": q = q[:, :3]
    if failure == "backend": kwargs["backend"] = "invented"
    if failure == "scale": kwargs["scale"] = math.nan
    with pytest.raises(ValueError): attend(q,k,v,qp,kp,**kwargs)


@pytest.mark.parametrize("arm", ARMS)
def test_fixture_end_to_end(arm):
    torch.manual_seed(3)
    model = Fixture().eval()
    rt = Runtime(model, Spec(arm, 1, 3)).freeze_backbone()
    ids = torch.tensor([[2,3,5,7,11,13,17]])
    with torch.no_grad():
        torch.testing.assert_close(rt(ids), model(ids).logits)
        if rt.update is not None: rt.update.up.weight.normal_(0,0.08)
        full = rt(ids)
        cached = torch.cat([rt(ids[:, i:i+1], use_cache=True) for i in range(ids.shape[1])], 1)
        torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-4)
        rt.reset()
        rt(ids[:, :3], use_cache=True)
        old = [(k.clone(),v.clone()) for k,v,_ in rt.cache]
        rt(ids[:, 3:5], use_cache=True)
        for (k,v,_),(oldk,oldv) in zip(rt.cache,old):
            assert torch.equal(k[:, :, :3],oldk) and torch.equal(v[:, :, :3],oldv)
        rt.reset()
        a = rt.generate(ids, max_new_tokens=3)
        b = rt.generate(ids, max_new_tokens=3, chunk_size=2)
        assert a == b and rt.position == 0 and all(x is None for x in rt.cache)


def test_fixture_full_qualification():
    result = qualify(Fixture().eval(), 1, 3)
    assert result["feedback/trainable_parameters"] == 52
    assert result["feedback/cache_max_error"] < 1e-5


@pytest.mark.parametrize("failure", ["dtype", "empty", "batch", "overlong", "gradcache", "invalid_token"])
def test_runtime_failure_resets_cache(failure):
    rt = Runtime(Fixture().eval(), Spec("feedback",1,3)).freeze_backbone()
    with torch.no_grad(): rt(torch.tensor([[1,2]]), use_cache=True)
    ids = torch.tensor([[3]])
    if failure == "dtype": ids = ids.float()
    if failure == "empty": ids = torch.empty(1,0,dtype=torch.long)
    if failure == "batch": ids = torch.ones(2,1,dtype=torch.long)
    if failure == "overlong": ids = torch.ones(1,256,dtype=torch.long)
    if failure == "invalid_token": ids = torch.tensor([[10000]])
    if failure == "gradcache":
        with pytest.raises(ValueError): rt(ids,use_cache=True)
    else:
        with torch.no_grad(), pytest.raises((ValueError,IndexError)): rt(ids,use_cache=True)
    assert rt.position == 0 and all(x is None for x in rt.cache)


def test_zero_vs_live_grad_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(9)
    model = Fixture().eval()
    rt = Runtime(model, Spec("feedback",1,3)).freeze_backbone()
    frozen = {n:p.detach().clone() for n,p in model.named_parameters()}
    protocol = {"learning_rate":0.01,"weight_decay":0,"gradient_clip":1.0}
    encoded = [({},[1,2,3],[4,5,6])]
    metrics = train_arm(rt, encoded, 8, 17, protocol, torch.device("cpu"))
    assert metrics["optimizer_steps"] == 8 and metrics["answer_tokens"] == 24
    assert all(torch.equal(frozen[n],p) for n,p in model.named_parameters())
    assert not torch.equal(rt.update.up.weight, torch.zeros_like(rt.update.up.weight))
    path = tmp_path/"adapter.pt"
    torch.save(rt.update.state_dict(),path)
    other = Runtime(model, Spec("feedback",1,3)).freeze_backbone()
    other.update.load_state_dict(torch.load(path,weights_only=True))
    ids = torch.tensor([[1,2,3]])
    torch.testing.assert_close(rt(ids),other(ids),atol=0,rtol=0)


def test_graph_oracle_and_disjoint_splits():
    train, dev = graph_rows("train",64,17), graph_rows("dev",48,17)
    assert train == graph_rows("train",64,17)
    assert train != graph_rows("train",64,18)
    vocab = []
    for rows in (train,dev):
        names = set()
        for r in rows:
            mapping = dict(r["edges"])
            assert len(mapping) == len(r["edges"])
            x = r["start"]
            for _ in range(r["hops"]): x = mapping[x]
            assert x == r["answer"]
            names.update(t for edge in r["edges"] for t in edge)
        vocab.append(names)
    assert not vocab[0] & vocab[1]
    assert {r["hops"] for r in train} == {1,2}
    assert {r["hops"] for r in dev} == {1,2,3,4}
    with pytest.raises(ValueError): graph_rows("sealed",1,17)


def records():
    return [{"arm":a,"seed":s,"id":str(i),"correct":a=="feedback"} for a in ARMS for s in (17,29,43) for i in range(4)]


def test_paired_summary():
    result = paired_summary(records())
    assert result["feedback_vs"]["independent"]["mean_delta"] == 1
    assert result["replication_unit"] == "training_seed" and not result["confirmatory"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "not_bool", "empty"])
def test_incomplete_evidence_rejected(failure):
    rows = records()
    if failure=="missing": rows.pop()
    if failure=="duplicate": rows.append(rows[0])
    if failure=="not_bool": rows[0]["correct"] = 1
    if failure=="empty": rows=[]
    with pytest.raises(ValueError): paired_summary(rows)


@pytest.mark.parametrize("failure", ["status", "stage", "binding", "sealed", "tamper"])
def test_gate_refuses_stale_receipts(failure):
    binding = {"code":"a", "model":"b"}
    receipt = {"stage":"qualify","status":"pass","binding":binding,"sealed_evaluation_authorized":False}
    receipt["receipt_sha256"] = digest(receipt)
    require_receipt(receipt,binding,"qualify")
    bad=copy.deepcopy(receipt)
    if failure=="status": bad["status"]="fail"
    if failure=="stage": bad["stage"]="smoke"
    if failure=="binding": bad["binding"]["model"]="changed"
    if failure=="sealed": bad["sealed_evaluation_authorized"]=True
    if failure=="tamper": bad["extra"]="tampered"
    with pytest.raises(ValueError): require_receipt(bad,binding,"qualify")


def test_native_hf_qualification():
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(901)
    c=Qwen3Config(hidden_size=32,intermediate_size=64,num_hidden_layers=3,
                   num_attention_heads=4,num_key_value_heads=2,head_dim=16,
                   vocab_size=73,max_position_embeddings=128,attention_dropout=0)
    c._attn_implementation="eager"
    model=Qwen3ForCausalLM(c).eval()
    assert qualify(model,1,3)["feedback/trainable_parameters"] == 100
    rt=Runtime(model,Spec("feedback",1,3)).freeze_backbone()
    result=train_arm(rt,[({},[1,2,3],[4,5])],5,17,
                     {"learning_rate":0.01,"weight_decay":0,"gradient_clip":1},torch.device("cpu"))
    assert result["optimizer_steps"]==5


@pytest.mark.parametrize("arm", ARMS)
def test_nonzero_rope_backend_and_gradient_equivalence(arm):
    q,k,v,qp,kp=tensors()
    angle=torch.rand(1,1,5,4)
    angle=torch.cat((angle,angle),-1)
    cos,sin=angle.cos(),angle.sin()
    k=rotate(k,cos,sin)
    up=Update(8,4,3)
    with torch.no_grad(): up.up.weight.normal_(0,0.2)
    other=copy.deepcopy(up)
    left=refined_read(q,k,v,qp,kp,cos,sin,Spec(arm,1,3,"eager"),up)
    right=refined_read(q,k,v,qp,kp,cos,sin,Spec(arm,1,3,"sdpa"),other)
    torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)
    if arm!="baseline":
        left.square().mean().backward();right.square().mean().backward()
        for a,b in zip(up.parameters(),other.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)


def test_control_is_not_feedback_and_both_reads_are_live():
    q,k,v,qp,kp=tensors()
    up=Update(8,4,3)
    with torch.no_grad(): up.up.weight.normal_(0,0.4)
    cos,sin=torch.ones(1,1,5,8),torch.zeros(1,1,5,8)
    baseline=attend(q,k,v,qp,kp)
    independent=refined_read(q,k,v,qp,kp,cos,sin,Spec("independent",1,3),up)
    feedback=refined_read(q,k,v,qp,kp,cos,sin,Spec("feedback",1,3),up)
    assert not torch.equal(independent,feedback)
    second=attend(q+up(q),k,v,qp,kp)
    torch.testing.assert_close(independent,0.5*(baseline+second))


def test_stopgrad_would_be_redundant_here():
    # A first read from a frozen prefix has no trainable ancestors in this design.
    q,k,v,qp,kp=tensors()
    first=attend(q,k,v,qp,kp)
    assert not first.requires_grad
    assert "stopgrad" not in ARMS


@pytest.mark.parametrize("arm", ARMS[1:])
def test_future_value_jacobian_zero(arm):
    q,k,v,qp,kp=tensors()
    v.requires_grad_()
    up=Update(8,4,3)
    with torch.no_grad(): up.up.weight.normal_(0,0.4)
    out=refined_read(q,k,v,qp,kp,torch.ones(1,1,5,8),torch.zeros(1,1,5,8),Spec(arm,1,3),up)
    out[:,:,:2].sum().backward()
    assert torch.equal(v.grad[:,:,2:],torch.zeros_like(v.grad[:,:,2:]))


@pytest.mark.parametrize("bad", [Spec("feedback",99,3),"hook","quantized","dtype","window","rope"])
def test_reject_unsupported_backbones(bad):
    model=Fixture().eval()
    spec=bad if isinstance(bad,Spec) else Spec("feedback",1,3)
    if bad=="hook": model.model.layers[0].register_forward_hook(lambda *args:None)
    if bad=="quantized": model.is_quantized=True
    if bad=="dtype": model.double()
    if bad=="window": model.config.use_sliding_window=True
    if bad=="rope": model.config.rope_scaling={"factor":2}
    with pytest.raises(ValueError): Runtime(model,spec)


@pytest.mark.parametrize("bad", [torch.empty(1,0,dtype=torch.long),torch.ones(2,3,dtype=torch.long),torch.ones(1,3)])
def test_invalid_generate_resets(bad):
    rt=Runtime(Fixture().eval(),Spec("feedback",1,3)).freeze_backbone()
    with pytest.raises(ValueError):rt.generate(bad)
    assert rt.position==0


def test_nonfinite_stream_is_rejected():
    rt=Runtime(Fixture().eval(),Spec("feedback",1,3)).freeze_backbone()
    with torch.no_grad():
        rt.update.up.weight.fill_(math.nan)
        with pytest.raises(FloatingPointError):rt(torch.tensor([[1,2]]),use_cache=True)
    assert rt.position==0 and all(x is None for x in rt.cache)


def test_answer_masks_and_no_truncation():
    class Tokenizer:
        eos_token_id=9
        def encode(self,text,add_special_tokens=False):return [1]*len(text)
    row={"prompt":"abc","answer":"x"}
    encoded=encode_rows(Tokenizer(),[row],6)
    assert len(encoded[0][1])==3 and encoded[0][2]==[1,1,9]
    with pytest.raises(ValueError):encode_rows(Tokenizer(),[row],5)
