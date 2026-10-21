import triton, triton.language as tl
@triton.jit
def k_rad2_scalar(RE, IM, ORE, OIM, K: tl.constexpr):
    bo = tl.program_id(0) * 8
    vr0 = tl.load(RE + bo + 0)
    vi0 = tl.load(IM + bo + 0)
    vr1 = tl.load(RE + bo + 4)
    vi1 = tl.load(IM + bo + 4)
    vr2 = tl.load(RE + bo + 2)
    vi2 = tl.load(IM + bo + 2)
    vr3 = tl.load(RE + bo + 6)
    vi3 = tl.load(IM + bo + 6)
    vr4 = tl.load(RE + bo + 1)
    vi4 = tl.load(IM + bo + 1)
    vr5 = tl.load(RE + bo + 5)
    vi5 = tl.load(IM + bo + 5)
    vr6 = tl.load(RE + bo + 3)
    vi6 = tl.load(IM + bo + 3)
    vr7 = tl.load(RE + bo + 7)
    vi7 = tl.load(IM + bo + 7)
    tr1 = vr1 * 1.00000000000000000 - vi1 * 0.00000000000000000
    ti1 = vr1 * 0.00000000000000000 + vi1 * 1.00000000000000000
    nr0 = vr0 + tr1
    ni0 = vi0 + ti1
    vr1 = vr0 - tr1
    vi1 = vi0 - ti1
    vr0 = nr0
    vi0 = ni0
    tr3 = vr3 * 1.00000000000000000 - vi3 * 0.00000000000000000
    ti3 = vr3 * 0.00000000000000000 + vi3 * 1.00000000000000000
    nr2 = vr2 + tr3
    ni2 = vi2 + ti3
    vr3 = vr2 - tr3
    vi3 = vi2 - ti3
    vr2 = nr2
    vi2 = ni2
    tr5 = vr5 * 1.00000000000000000 - vi5 * 0.00000000000000000
    ti5 = vr5 * 0.00000000000000000 + vi5 * 1.00000000000000000
    nr4 = vr4 + tr5
    ni4 = vi4 + ti5
    vr5 = vr4 - tr5
    vi5 = vi4 - ti5
    vr4 = nr4
    vi4 = ni4
    tr7 = vr7 * 1.00000000000000000 - vi7 * 0.00000000000000000
    ti7 = vr7 * 0.00000000000000000 + vi7 * 1.00000000000000000
    nr6 = vr6 + tr7
    ni6 = vi6 + ti7
    vr7 = vr6 - tr7
    vi7 = vi6 - ti7
    vr6 = nr6
    vi6 = ni6
    tr2 = vr2 * 1.00000000000000000 - vi2 * 0.00000000000000000
    ti2 = vr2 * 0.00000000000000000 + vi2 * 1.00000000000000000
    nr0 = vr0 + tr2
    ni0 = vi0 + ti2
    vr2 = vr0 - tr2
    vi2 = vi0 - ti2
    vr0 = nr0
    vi0 = ni0
    tr3 = vr3 * 0.00000000000000006 - vi3 * -1.00000000000000000
    ti3 = vr3 * -1.00000000000000000 + vi3 * 0.00000000000000006
    nr1 = vr1 + tr3
    ni1 = vi1 + ti3
    vr3 = vr1 - tr3
    vi3 = vi1 - ti3
    vr1 = nr1
    vi1 = ni1
    tr6 = vr6 * 1.00000000000000000 - vi6 * 0.00000000000000000
    ti6 = vr6 * 0.00000000000000000 + vi6 * 1.00000000000000000
    nr4 = vr4 + tr6
    ni4 = vi4 + ti6
    vr6 = vr4 - tr6
    vi6 = vi4 - ti6
    vr4 = nr4
    vi4 = ni4
    tr7 = vr7 * 0.00000000000000006 - vi7 * -1.00000000000000000
    ti7 = vr7 * -1.00000000000000000 + vi7 * 0.00000000000000006
    nr5 = vr5 + tr7
    ni5 = vi5 + ti7
    vr7 = vr5 - tr7
    vi7 = vi5 - ti7
    vr5 = nr5
    vi5 = ni5
    tr4 = vr4 * 1.00000000000000000 - vi4 * 0.00000000000000000
    ti4 = vr4 * 0.00000000000000000 + vi4 * 1.00000000000000000
    nr0 = vr0 + tr4
    ni0 = vi0 + ti4
    vr4 = vr0 - tr4
    vi4 = vi0 - ti4
    vr0 = nr0
    vi0 = ni0
    tr5 = vr5 * 0.70710678118654757 - vi5 * -0.70710678118654746
    ti5 = vr5 * -0.70710678118654746 + vi5 * 0.70710678118654757
    nr1 = vr1 + tr5
    ni1 = vi1 + ti5
    vr5 = vr1 - tr5
    vi5 = vi1 - ti5
    vr1 = nr1
    vi1 = ni1
    tr6 = vr6 * 0.00000000000000006 - vi6 * -1.00000000000000000
    ti6 = vr6 * -1.00000000000000000 + vi6 * 0.00000000000000006
    nr2 = vr2 + tr6
    ni2 = vi2 + ti6
    vr6 = vr2 - tr6
    vi6 = vi2 - ti6
    vr2 = nr2
    vi2 = ni2
    tr7 = vr7 * -0.70710678118654746 - vi7 * -0.70710678118654757
    ti7 = vr7 * -0.70710678118654757 + vi7 * -0.70710678118654746
    nr3 = vr3 + tr7
    ni3 = vi3 + ti7
    vr7 = vr3 - tr7
    vi7 = vi3 - ti7
    vr3 = nr3
    vi3 = ni3
    tl.store(ORE + bo + 0, vr0)
    tl.store(OIM + bo + 0, vi0)
    tl.store(ORE + bo + 1, vr1)
    tl.store(OIM + bo + 1, vi1)
    tl.store(ORE + bo + 2, vr2)
    tl.store(OIM + bo + 2, vi2)
    tl.store(ORE + bo + 3, vr3)
    tl.store(OIM + bo + 3, vi3)
    tl.store(ORE + bo + 4, vr4)
    tl.store(OIM + bo + 4, vi4)
    tl.store(ORE + bo + 5, vr5)
    tl.store(OIM + bo + 5, vi5)
    tl.store(ORE + bo + 6, vr6)
    tl.store(OIM + bo + 6, vi6)
    tl.store(ORE + bo + 7, vr7)
    tl.store(OIM + bo + 7, vi7)
