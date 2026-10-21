import triton, triton.language as tl
@triton.jit
def k_rad2_scalar(RE, IM, ORE, OIM, K: tl.constexpr):
    bo = tl.program_id(0) * 16
    vr0 = tl.load(RE + bo + 0)
    vi0 = tl.load(IM + bo + 0)
    vr1 = tl.load(RE + bo + 8)
    vi1 = tl.load(IM + bo + 8)
    vr2 = tl.load(RE + bo + 4)
    vi2 = tl.load(IM + bo + 4)
    vr3 = tl.load(RE + bo + 12)
    vi3 = tl.load(IM + bo + 12)
    vr4 = tl.load(RE + bo + 2)
    vi4 = tl.load(IM + bo + 2)
    vr5 = tl.load(RE + bo + 10)
    vi5 = tl.load(IM + bo + 10)
    vr6 = tl.load(RE + bo + 6)
    vi6 = tl.load(IM + bo + 6)
    vr7 = tl.load(RE + bo + 14)
    vi7 = tl.load(IM + bo + 14)
    vr8 = tl.load(RE + bo + 1)
    vi8 = tl.load(IM + bo + 1)
    vr9 = tl.load(RE + bo + 9)
    vi9 = tl.load(IM + bo + 9)
    vr10 = tl.load(RE + bo + 5)
    vi10 = tl.load(IM + bo + 5)
    vr11 = tl.load(RE + bo + 13)
    vi11 = tl.load(IM + bo + 13)
    vr12 = tl.load(RE + bo + 3)
    vi12 = tl.load(IM + bo + 3)
    vr13 = tl.load(RE + bo + 11)
    vi13 = tl.load(IM + bo + 11)
    vr14 = tl.load(RE + bo + 7)
    vi14 = tl.load(IM + bo + 7)
    vr15 = tl.load(RE + bo + 15)
    vi15 = tl.load(IM + bo + 15)
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
    tr9 = vr9 * 1.00000000000000000 - vi9 * 0.00000000000000000
    ti9 = vr9 * 0.00000000000000000 + vi9 * 1.00000000000000000
    nr8 = vr8 + tr9
    ni8 = vi8 + ti9
    vr9 = vr8 - tr9
    vi9 = vi8 - ti9
    vr8 = nr8
    vi8 = ni8
    tr11 = vr11 * 1.00000000000000000 - vi11 * 0.00000000000000000
    ti11 = vr11 * 0.00000000000000000 + vi11 * 1.00000000000000000
    nr10 = vr10 + tr11
    ni10 = vi10 + ti11
    vr11 = vr10 - tr11
    vi11 = vi10 - ti11
    vr10 = nr10
    vi10 = ni10
    tr13 = vr13 * 1.00000000000000000 - vi13 * 0.00000000000000000
    ti13 = vr13 * 0.00000000000000000 + vi13 * 1.00000000000000000
    nr12 = vr12 + tr13
    ni12 = vi12 + ti13
    vr13 = vr12 - tr13
    vi13 = vi12 - ti13
    vr12 = nr12
    vi12 = ni12
    tr15 = vr15 * 1.00000000000000000 - vi15 * 0.00000000000000000
    ti15 = vr15 * 0.00000000000000000 + vi15 * 1.00000000000000000
    nr14 = vr14 + tr15
    ni14 = vi14 + ti15
    vr15 = vr14 - tr15
    vi15 = vi14 - ti15
    vr14 = nr14
    vi14 = ni14
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
    tr10 = vr10 * 1.00000000000000000 - vi10 * 0.00000000000000000
    ti10 = vr10 * 0.00000000000000000 + vi10 * 1.00000000000000000
    nr8 = vr8 + tr10
    ni8 = vi8 + ti10
    vr10 = vr8 - tr10
    vi10 = vi8 - ti10
    vr8 = nr8
    vi8 = ni8
    tr11 = vr11 * 0.00000000000000006 - vi11 * -1.00000000000000000
    ti11 = vr11 * -1.00000000000000000 + vi11 * 0.00000000000000006
    nr9 = vr9 + tr11
    ni9 = vi9 + ti11
    vr11 = vr9 - tr11
    vi11 = vi9 - ti11
    vr9 = nr9
    vi9 = ni9
    tr14 = vr14 * 1.00000000000000000 - vi14 * 0.00000000000000000
    ti14 = vr14 * 0.00000000000000000 + vi14 * 1.00000000000000000
    nr12 = vr12 + tr14
    ni12 = vi12 + ti14
    vr14 = vr12 - tr14
    vi14 = vi12 - ti14
    vr12 = nr12
    vi12 = ni12
    tr15 = vr15 * 0.00000000000000006 - vi15 * -1.00000000000000000
    ti15 = vr15 * -1.00000000000000000 + vi15 * 0.00000000000000006
    nr13 = vr13 + tr15
    ni13 = vi13 + ti15
    vr15 = vr13 - tr15
    vi15 = vi13 - ti15
    vr13 = nr13
    vi13 = ni13
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
    tr12 = vr12 * 1.00000000000000000 - vi12 * 0.00000000000000000
    ti12 = vr12 * 0.00000000000000000 + vi12 * 1.00000000000000000
    nr8 = vr8 + tr12
    ni8 = vi8 + ti12
    vr12 = vr8 - tr12
    vi12 = vi8 - ti12
    vr8 = nr8
    vi8 = ni8
    tr13 = vr13 * 0.70710678118654757 - vi13 * -0.70710678118654746
    ti13 = vr13 * -0.70710678118654746 + vi13 * 0.70710678118654757
    nr9 = vr9 + tr13
    ni9 = vi9 + ti13
    vr13 = vr9 - tr13
    vi13 = vi9 - ti13
    vr9 = nr9
    vi9 = ni9
    tr14 = vr14 * 0.00000000000000006 - vi14 * -1.00000000000000000
    ti14 = vr14 * -1.00000000000000000 + vi14 * 0.00000000000000006
    nr10 = vr10 + tr14
    ni10 = vi10 + ti14
    vr14 = vr10 - tr14
    vi14 = vi10 - ti14
    vr10 = nr10
    vi10 = ni10
    tr15 = vr15 * -0.70710678118654746 - vi15 * -0.70710678118654757
    ti15 = vr15 * -0.70710678118654757 + vi15 * -0.70710678118654746
    nr11 = vr11 + tr15
    ni11 = vi11 + ti15
    vr15 = vr11 - tr15
    vi15 = vi11 - ti15
    vr11 = nr11
    vi11 = ni11
    tr8 = vr8 * 1.00000000000000000 - vi8 * 0.00000000000000000
    ti8 = vr8 * 0.00000000000000000 + vi8 * 1.00000000000000000
    nr0 = vr0 + tr8
    ni0 = vi0 + ti8
    vr8 = vr0 - tr8
    vi8 = vi0 - ti8
    vr0 = nr0
    vi0 = ni0
    tr9 = vr9 * 0.92387953251128674 - vi9 * -0.38268343236508978
    ti9 = vr9 * -0.38268343236508978 + vi9 * 0.92387953251128674
    nr1 = vr1 + tr9
    ni1 = vi1 + ti9
    vr9 = vr1 - tr9
    vi9 = vi1 - ti9
    vr1 = nr1
    vi1 = ni1
    tr10 = vr10 * 0.70710678118654757 - vi10 * -0.70710678118654746
    ti10 = vr10 * -0.70710678118654746 + vi10 * 0.70710678118654757
    nr2 = vr2 + tr10
    ni2 = vi2 + ti10
    vr10 = vr2 - tr10
    vi10 = vi2 - ti10
    vr2 = nr2
    vi2 = ni2
    tr11 = vr11 * 0.38268343236508984 - vi11 * -0.92387953251128674
    ti11 = vr11 * -0.92387953251128674 + vi11 * 0.38268343236508984
    nr3 = vr3 + tr11
    ni3 = vi3 + ti11
    vr11 = vr3 - tr11
    vi11 = vi3 - ti11
    vr3 = nr3
    vi3 = ni3
    tr12 = vr12 * 0.00000000000000006 - vi12 * -1.00000000000000000
    ti12 = vr12 * -1.00000000000000000 + vi12 * 0.00000000000000006
    nr4 = vr4 + tr12
    ni4 = vi4 + ti12
    vr12 = vr4 - tr12
    vi12 = vi4 - ti12
    vr4 = nr4
    vi4 = ni4
    tr13 = vr13 * -0.38268343236508973 - vi13 * -0.92387953251128674
    ti13 = vr13 * -0.92387953251128674 + vi13 * -0.38268343236508973
    nr5 = vr5 + tr13
    ni5 = vi5 + ti13
    vr13 = vr5 - tr13
    vi13 = vi5 - ti13
    vr5 = nr5
    vi5 = ni5
    tr14 = vr14 * -0.70710678118654746 - vi14 * -0.70710678118654757
    ti14 = vr14 * -0.70710678118654757 + vi14 * -0.70710678118654746
    nr6 = vr6 + tr14
    ni6 = vi6 + ti14
    vr14 = vr6 - tr14
    vi14 = vi6 - ti14
    vr6 = nr6
    vi6 = ni6
    tr15 = vr15 * -0.92387953251128674 - vi15 * -0.38268343236508989
    ti15 = vr15 * -0.38268343236508989 + vi15 * -0.92387953251128674
    nr7 = vr7 + tr15
    ni7 = vi7 + ti15
    vr15 = vr7 - tr15
    vi15 = vi7 - ti15
    vr7 = nr7
    vi7 = ni7
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
    tl.store(ORE + bo + 8, vr8)
    tl.store(OIM + bo + 8, vi8)
    tl.store(ORE + bo + 9, vr9)
    tl.store(OIM + bo + 9, vi9)
    tl.store(ORE + bo + 10, vr10)
    tl.store(OIM + bo + 10, vi10)
    tl.store(ORE + bo + 11, vr11)
    tl.store(OIM + bo + 11, vi11)
    tl.store(ORE + bo + 12, vr12)
    tl.store(OIM + bo + 12, vi12)
    tl.store(ORE + bo + 13, vr13)
    tl.store(OIM + bo + 13, vi13)
    tl.store(ORE + bo + 14, vr14)
    tl.store(OIM + bo + 14, vi14)
    tl.store(ORE + bo + 15, vr15)
    tl.store(OIM + bo + 15, vi15)
