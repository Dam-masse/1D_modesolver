# python solver based on graphical solution by Pollock equation
#
# DM 16/12/2019

# importing usefull libraries
import numpy as np
import matplotlib.pyplot as plt

""" waveguide structure definition of names 
n_c                                  cladding 
____________________________________ 
          ^
n_f       h                          waveguiding layer 
__________v_________________________

n_s                                  substrate layer 


In the following the subscript c,f,s will be referenced as the quantities refered to the correspondent layer 
n refractive inde
h thickness of the waveguiding layer 
lambda0  vacuum wavelength
k0       vacuum wavevector
"""


h = 1  # thickness of the waveguide in micrometers
lambda0 = 1.55  # wavelength in microns
k0 = (2 * np.pi) / lambda0  # vacuuum wavevector
n_c = 1  # refractive index of the cladding
n_f = 1.5  # waveguide refractive index of the waveguide layer
n_s = 1.3  # substrate refractive inde of the sustrate

wg_vector = [
    n_c,
    n_f,
    n_s,
    h,
]  # vector that will be feed into the functions and that rappresent the waveguide it is made in order to keep the description as general as possible
# definition of function that will be used in the code


def mode_function_difference(n_eff, pol="TE", wg=wg_vector, k=k0):
    """This function takes as input a value of effective index, then evaluate the corresponding wavevectors.
    after selecting the espession to use based on the polarization, it compute the difference that correspond to
    lhs-rhs of the equation in the pollock book

    n_eff double is the effective index where i want to calculate the difference

    pol  str is a strig that indicates the polarization

    wg is a vector that rappresents the waveguide define by [n_c,n_f,n_s,h]

    k is the wavevector in vacuum
    """

    beta = n_eff * k
    # the beta is what is commonly used to indentify the z component of the wavevector, give z as the propagation vector
    kf = np.sqrt((k ** 2) * (wg[1] ** 2) - (beta ** 2))
    # kf is the x component of the wavevector, this is refered to the part internal to the waveguide
    g_c = np.sqrt((-(k ** 2)) * (wg[0] ** 2) + (beta ** 2))
    # g_c is the x component of the wavevector, this is refered to the part in the cladding layer
    g_s = np.sqrt(((-(k ** 2))) * (wg[2] ** 2) + (beta ** 2))
    # g_s is the x component of the wavevector, this is refered to the part in the substrate
    if pol == "TE":
        difference = np.tan(wg[3] * kf) - (g_c + g_s) / (kf * (1 - g_c * g_s / kf ** 2))
        # in TE case
    else:
        difference = np.tan(wg[3] * kf) - (
            (
                kf
                * (
                    ((wg[1] ** 2 / wg[2] ** 2) * g_s)
                    + ((wg[1] ** 2 / wg[0] ** 2) * g_c)
                )
            )
            / (kf ** 2 - (wg[1] ** 4 / (wg[0] ** 2 * wg[2] ** 2)) * g_c * g_s)
        )
        # in TM case
    return difference


def plot_from_neff(N, polar, wg=wg_vector, k=k0, save_bool=True):
    """this function plot the electric field y component or the H field x component, based no the effective index and the waveguide index profile

    N double is the effective index that has been found by the alghoritm

    polar is the polarization of the mode

    wg is a vector that rapresents the waveguide defined by [n_c,n_f,n_s,h]

    k is the wavevector in vacuum

    save_bool is a boolean variable, if it is true the programm will save the figures
    """

    g_c = k * np.sqrt(N ** 2 - wg[0] ** 2)
    # g_c is the x component of the wavevector, this is refered to the part in the cladding layer
    kf = k * np.sqrt(wg[1] ** 2 - N ** 2)
    # kf is the x component of the wavevector, this is refered to the part internal to the waveguide
    g_s = k * np.sqrt(N ** 2 - wg[2] ** 2)
    # g_s is the x component of the wavevector, this is refered to the part in the substrate

    # builidng position vectors
    x_f = np.linspace(-wg[3], 0, 100)  # waveguide layer position vector
    x_c = np.linspace(0, 2 * wg[3], 100)
    x_s = np.linspace(-wg[3], -2 * wg[3], 100)
    A2 = 1  # normalization constant, to normalize in power this has ho be equal to the inverse of the integrall of the power of the fields
    E_c = A2 * np.exp(-g_c * x_c)  # gamma c is g_c
    if polar == "TE":
        E_f = A2 * (
            np.cos(kf * x_f) - (g_c / kf) * np.sin(kf * x_f)
        )  # TE    equations taken from Pollock fundamentals
        E_s = (
            A2
            * (np.cos(kf * wg[3]) + (g_c / kf) * np.sin(kf * wg[3]))
            * np.exp(g_s * (x_s + wg[3]))
        )
    elif polar == "TM":
        E_f = A2 * (
            np.cos(kf * x_f) - (wg[1] ** 2 / wg[0] ** 2 * (g_c / kf)) * np.sin(kf * x_f)
        )  # TM
        E_s = (
            A2
            * (
                np.cos(kf * wg[3])
                + ((wg[1] ** 2 / wg[0] ** 2 * g_c / kf)) * np.sin(kf * wg[3])
            )
            * np.exp(g_s * (x_s + wg[3]))
        )
    E_f = np.flip(E_f)
    x_f = np.flip(x_f)
    E_c = np.flip(E_c)
    x_c = np.flip(x_c)
    field = np.concatenate(
        (
            E_c,
            E_f,
            E_s,
        )
    )
    # concatenation of the relevant field arrays
    position = np.concatenate((x_c, x_f, x_s))  # concatenation of the position arrays
    # waveguide walls.
    yw = (0, 0)
    yw1 = (-wg[3], -wg[3])
    xw = (-3, 3)
    # plotting
    plt.figure()
    # plt.plot(x_c,E_c,'r')
    # plt.plot(x_f,E_f,'r')
    # plt.plot(x_s,E_s,'r')
    plt.plot(position, field)
    plt.xlim(-2 * wg[3], wg[3])
    plt.plot(yw, xw, "b")
    plt.plot(yw1, xw, "b")
    if polar == "TE":
        plt.title("$E_y$ Field Plot $N eff=$" + str(N))
    elif polar == "TM":
        plt.title("$H_y$ Field Plot $N eff=$" + str(N))
    plt.ylabel("Field Intensity")
    plt.xlabel("Waveguide Profile")
    if save_bool:  # if true the program will save the figures as png
        plt.savefig("./mode_" + polar + str(N) + ".png", format="png")


if __name__ == "__main__":
    """main part of the code
    in this part we calculate the modes using a bisection methon on the mode_function_difference function"""

    N = 1000  # dimension of  vector number of point for an initial estimation
    polarization = ["TE", "TM"]  # vector containig the 2 posible polarizations
    for pol in polarization:  # we circle throught the polarizations
        n_eff = np.linspace(
            n_f - 0.000000001, 1, N
        )  # creating a vector of n_eff that will be tested againt the funciton
        diff = mode_function_difference(n_eff, pol)
        # in this case is not necessary to use the wg an k input of the function
        i_list = []  # list of the indexes that exhibit a change in the sign
        n_eff_solution_list = []
        # list of the effective indexe that are solution of mode_function_differenc(n_eff)=0
        for i in range(N - 1):
            if (diff[i] < 0 & 0 < diff[i + 1]) or (
                diff[i + 1] < 0 & 0 < diff[i]
            ):  # finding pairs of points on opposite side of 0
                if abs(diff[i] - diff[i + 1]) < 1:
                    # ignoring points related to sign swaps from the asymptotes
                    i_list.append(i)  # adding the selected point to the list

        for index in i_list:  # starting a search using the previous selected indexes
            not_found = True  # boolean that indicate if the n_eff that solve the problem has been found
            # if true we append this value to the n_eff_solution_list and stop the while loop
            a_n = n_eff[index]  # this are the extremes of a bisection method
            b_n = n_eff[index + 1]
            epsilon = (
                0.000000001  # precision require to the bisection method to converge
            )
            while not_found:
                m_n = (a_n + b_n) / 2
                f_m_n = mode_function_difference(m_n, pol)
                if mode_function_difference(a_n, pol) * f_m_n < 0:  # bisection method
                    a_n = a_n
                    b_n = m_n
                else:
                    a_n = m_n
                    b_n = b_n
                if abs(f_m_n) < epsilon:
                    print("Found exact solution.")
                    not_found = False
                    n_eff_solution_list.append(m_n)
        for n_effective in n_eff_solution_list:
            # now we use the found effective inde in the plot function
            plot_from_neff(n_effective, pol)
    plt.show()  # show the final plots
