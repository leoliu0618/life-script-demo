def diagnose(answers: list[dict]):
    """
    answers: [{'q': 'inj_x', 'a': '...'}, ...]
    这里只做演示：统计包含 inj / driver 的条数，再给一个 winner / non-winner 标签
    """
    inj_cnt = sum(1 for x in answers if 'inj' in x['q'])
    driver_cnt = sum(1 for x in answers if 'driver' in x['q'])
    summary = "Winner" if driver_cnt > inj_cnt else "Non-winner"
    return {"summary": summary, "inj_cnt": inj_cnt, "driver_cnt": driver_cnt}
