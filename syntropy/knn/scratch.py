import jpype as jp
import numpy as np
import pandas 
import sys
# Our python data file readers are a bit of a hack, python users will do better on this:
sys.path.append("/home/thosvarley/.local/share/JIDT/demos/python")
import readFloatsFile

if (not jp.isJVMStarted()):
    # Add JIDT jar library to the path
    jarLocation = "/home/thosvarley/.local/share/JIDT/infodynamics.jar"
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings=True)

# %%
def jidt_oinfo(data : np.ndarray) -> float:
    """
    Wrapper function for the Kraskov JIDT O-information.
    """
    oinfo_calc_class = jp.JPackage("infodynamics.measures.continuous.kraskov").OInfoCalculatorKraskov
    oinfo_calc = oinfo_calc_class()
    
    oinfo_calc.initialise(data.shape[1])
    
    oinfo_calc.setProperty("k","1") 
    oinfo_calc.setProperty("NORMALISE", "false")
    oinfo_calc.setProperty("NOISE_LEVEL_TO_ADD", "0")
    
    oinfo_calc.setObservations(jp.JArray(jp.JDouble, 2)(data.tolist()))
    
    return oinfo_calc.computeAverageLocalOfObservations()

def jidt_sinfo(data : np.ndarray) -> float:
    """
    Wrapper function for the Kraskov JIDT O-information.
    """
    sinfo_calc_class = jp.JPackage("infodynamics.measures.continuous.kraskov").SInfoCalculatorKraskov
    sinfo_calc = sinfo_calc_class()
    
    sinfo_calc.initialise(data.shape[1])
    
    sinfo_calc.setProperty("k","1") 
    sinfo_calc.setProperty("NORMALISE", "false")
    sinfo_calc.setProperty("NOISE_LEVEL_TO_ADD", "0")
    
    sinfo_calc.setObservations(jp.JArray(jp.JDouble, 2)(data.tolist()))
    
    return sinfo_calc.computeAverageLocalOfObservations()

def jidt_total_correlation(data : np.ndarray) -> float:

    tc_calc_class = jp.JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov1
    tc_calc = tc_calc_class()
    tc_calc.initialise(data.shape[1])

    tc_calc.setProperty("k","1") 
    tc_calc.setProperty("NOISE_LEVEL_TO_ADD", "0")
    #oinfo_calc.setProperty("NORM_TYPE", "EUCLIDEAN") # Uncomment to change from default of the maximum  norm
    tc_calc.setProperty("NORMALISE", "false")
    tc_calc.setObservations(jp.JArray(jp.JDouble, 2)(data.tolist()))

    return tc_calc.computeAverageLocalOfObservations()

def jidt_dual_total_correlation(data : np.ndarray) -> float:

    dtc_calc_class = jp.JPackage("infodynamics.measures.continuous.kraskov").DualTotalCorrelationCalculatorKraskov
    dtc_calc = dtc_calc_class()
    dtc_calc.initialise(data.shape[1])

    dtc_calc.setProperty("k","1") 
    #oinfo_calc.setProperty("NORM_TYPE", "EUCLIDEAN") # Uncomment to change from default of the maximum  norm
    dtc_calc.setProperty("NORMALISE", "false")
    dtc_calc.setProperty("NOISE_LEVEL_TO_ADD", "0")
    dtc_calc.setObservations(jp.JArray(jp.JDouble, 2)(data.tolist()))

    return dtc_calc.computeAverageLocalOfObservations()

# %%

# 0. Load/prepare the data:
data = pandas.read_csv("/home/thosvarley/Code/syntropy/tests/stack.csv", header=None).values.T

result = jidt_sinfo(data.T)
