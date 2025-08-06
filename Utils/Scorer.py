import numpy

def score1(time1):
    if len(time1):
        avg_time1 = sum(time1) / len(time1)
        avg_time1 = avg_time1 / 60
        if avg_time1 <= 5:
            return 20
        elif avg_time1 > 5 and avg_time1 <= 10:
            return 10
        else:
            return 0
    else:
        return 0
    
def score2(time2):
    if len(time2):
        avg_time2 = sum(time2) / len(time2)
        avg_time2 = avg_time2 / 60
        if avg_time2 <= 10:
            return 20
        elif avg_time2 > 10 and avg_time2 <= 15:
            return 10
        else:
            return 0
    else:
        return 0