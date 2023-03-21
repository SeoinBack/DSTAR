import numpy as np

def activity(co, h, oh, eU, product):
    if product == 'formate':
        rxn1 = h + 0.178 - 0 + eU
        rxn2 = 1.1757*oh - h - 0.916 - 0
        rxn3 = 0.32 - 1.1757*oh + 0.738 + eU
    
        return max(rxn1,rxn3,rxn2)
    
    elif product == 'h2':
        rxn1 = h + 0.178 - 0 + eU
        rxn2 = 0 - h - 0.178 + eU
        
        return max(rxn1,rxn2)
    
    elif product == 'c1':
        rxn1 = 0.6014*co + 0.3438*oh + 0.5289 - 0 + eU
        
        if co + 0.32 - 0 <= 0:
            rxn2 = 0.3986*co - 0.3438*oh - 0.2089 + eU
            rxn3 = 0.5628*co + 1.6066 + eU
        else:
            rxn2 = 0 - 0.6014*co - 0.3438*oh - 0.5289 + eU
            rxn3 = 1.5628*co + 1.9266 + eU
        
        return(max(rxn1,rxn2,rxn3))
    
    elif product == 'co':
        rxn1 = 0.6014*co + 0.3438*oh + 0.5289- 0 + eU

        if co + 0.32 <= 0:
            if 0.563*co + 0.857 + eU <0:
                rxn2 = 0.3986*co - 0.3438*oh - 0.2089 + eU
                rxn3 = 0.5628*co + 1.6066 + eU
                rxn4 = -100
            else:
                rxn2 = 0.3986*co - 0.3438*oh - 0.2089 + eU
                rxn3 = 0 - 0.32 - co
                rxn4 = -100
        else:
            if 1.563*co + 1.222 + eU < 0:
                rxn2 = 0.3986*co - 0.3438*oh - 0.2089 + eU
                rxn3 = 0 - 0.32 - co
                rxn4 = 1.5628*co + 1.9266 + eU
            else:
                rxn2 = 0.3986*co - 0.3438*oh - 0.2089 + eU
                rxn3 = 0 - 0.32 - co
                rxn4 = -100
                
        return max(rxn1,rxn2,rxn3,rxn4)
        
    else:
        return np.nan
        
def boundary_condition(co, h, oh, eU):    
    if (h - 0.22 - eU > 0):
        if (0.601*co - 0.832*oh + h + 1.445 + eU > 0) & (h + 0.178 + eU < 0):
            return 'formate'
        elif (0.601*co - 0.832*oh + h + 1.445 + eU < 0):
            if (-0.507*co - h - 0.89 < 0):
                if (co <= -0.32) & (0.563*co + 0.857 + eU < 0):
                    return 'c1'
                elif (co > -0.32) & (1.563*co + 1.222 + eU < 0):
                    return 'c1'
                elif (co <= -0.32) & (0.563*co + 0.857 + eU > 0):
                    return 'co'
                elif (co > -0.32) & (1.563*co + 1.222 + eU > 0):
                    return 'co'
            elif (-0.507*co - h - 0.89 > 0) & (h + 0.178 + eU < 0):
                return 'h2'
    