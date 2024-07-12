/** active model B+
 * 
 * This is a wrapped version of E. Tjhung's implementation:
 * https://github.com/elsentjhung/active-model-B-plus
 */

/* preprocessor */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include <numbers>
static constexpr double pi = std::numbers::pi;

#include "math_primitives.h"

struct TjhungIntegrator
{
    static constexpr double phi_init = -0.4; // initial density

    const std::size_t Nx, Ny;
    const double dx, dy, dt;
    const double Lx, Ly;
    const double A, K, lambda, zeta, D;
    int timestep;

    Field phi;  // density field 
    Field dphi;  // changes in phi

    Field Jx, Jy;  // total current
    Field f;  // free energy density

    Field mueq;  // equilibrium chemical potential
    Field muact;  // active chemical potential
    Field Jxeq, Jyeq;  // equilibrium currents
    Field Jxact, Jyact;  // active currents
    Field Lambdax;  // noise current field
    Field Lambday;

    std::vector<int> iupa, idwna;
    std::vector<int> jupa, jdwna;


    inline TjhungIntegrator(std::size_t Nx, std::size_t Ny,
                            double dt, double dx, double dy,
                            double A, double K, double lambda, double zeta, double D)
        : Nx(Nx), Ny(Ny), dx(dx), dy(dy), dt(dt),
        Lx(Nx*dx), Ly(Ny*dy), A(A), K(K), lambda(lambda), zeta(zeta), D(D),
        timestep(0),
        phi(Nx, Ny),
        dphi(Nx, Ny),
        Jx(Nx, Ny),
        Jy(Nx, Ny),
        f(Nx, Ny),
        mueq(Nx, Ny),
        muact(Nx, Ny),
        Jxeq(Nx, Ny),
        Jyeq(Nx, Ny),
        Jxact(Nx, Ny),
        Jyact(Nx, Ny),
        Lambdax(Nx, Ny),
        Lambday(Nx, Ny),
        iupa(Nx), idwna(Nx), jupa(Ny), jdwna(Ny)
    {
        srand48((unsigned int) std::time(nullptr));
        initialize();
    }

    inline void run(int Nt)
    {
        for (int nt = 0; nt < Nt; nt++)
        {
            calculate_dphi();
            update_phi();                
        }
    }

    // get random number from normal distribution
    inline double gaussian_rand(void) {
        static int iset = 0;
        static double gset;
        double fac, rsq, v1, v2;
    
        if (iset == 0) {
            do {
                v1 = 2.0*drand48()-1.0;
                v2 = 2.0*drand48()-1.0;
                rsq = v1*v1 + v2*v2;
            } while (rsq >= 1.0 || rsq == 0.0);
            fac = sqrt(-2.0*log(rsq)/rsq);
        
            gset = v1*fac;
            iset = 1;
            return v2*fac;
        } else {
            iset = 0;
            return gset;
        }
    }

    // second order numerical derivative
    inline double diff_x(FieldRef arr, int i, int j) {
        double dxphi;
        int iup = iupa[i];
        int idwn = idwna[i];
        int jup = jupa[j];
        int jdwn = jdwna[j];

        dxphi = ((arr(iup,jup)  - arr(idwn,jup))*0.1 + 
                (arr(iup,j)    - arr(idwn,j))*0.3   +
                (arr(iup,jdwn) - arr(idwn,jdwn))*0.1)/dx;

        return dxphi;
    }

    inline double diff_y(FieldRef arr, int i, int j) {
        double dyphi;
        int iup = iupa[i];
        int idwn = idwna[i];
        int jup = jupa[j];
        int jdwn = jdwna[j];

        dyphi = ((arr(iup,jup)  - arr(iup,jdwn))*0.1 + 
                (arr(i,jup)    - arr(i,jdwn))*0.3   +
                (arr(idwn,jup) - arr(idwn,jdwn))*0.1)/dy;

        return dyphi;
    }

    inline double laplacian(FieldRef arr, int i, int j) {
        double laplacianphi;
        int iup = iupa[i];
        int idwn = idwna[i];
        int jup = jupa[j];
        int jdwn = jdwna[j];

        laplacianphi = ((-0.5*arr(idwn,jup)  + 2.0*arr(i,jup)  - 0.5*arr(iup,jup)) + 
                        ( 2.0*arr(idwn,j)    - 6.0*arr(i,j)    + 2.0*arr(iup,j))   +
                        (-0.5*arr(idwn,jdwn) + 2.0*arr(i,jdwn) - 0.5*arr(iup,jdwn)))/(dx*dy);

        return laplacianphi;
    }

    // eighth order numerical derivative
    inline double diff_x_8(FieldRef arr, int i, int j) {
        double dxphi;
        int iup = iupa[i];
        int iup2 = iupa[iupa[i]];
        int iup3 = iupa[iupa[iupa[i]]];
        int iup4 = iupa[iupa[iupa[iupa[i]]]];
        int idwn = idwna[i];
        int idwn2 = idwna[idwna[i]];
        int idwn3 = idwna[idwna[idwna[i]]];
        int idwn4 = idwna[idwna[idwna[idwna[i]]]];

        dxphi = (-arr(iup4,j)/280.0  + 4.0*arr(iup3,j)/105.0  - arr(iup2,j)/5.0  + 4.0*arr(iup,j)/5.0  + 
                arr(idwn4,j)/280.0 - 4.0*arr(idwn3,j)/105.0 + arr(idwn2,j)/5.0 - 4.0*arr(idwn,j)/5.0)/dx;

        return dxphi;
    }

    inline double diff_y_8(FieldRef arr, int i, int j) {
        double dyphi;
        int jup = jupa[j];
        int jup2 = jupa[jupa[j]];
        int jup3 = jupa[jupa[jupa[j]]];
        int jup4 = jupa[jupa[jupa[jupa[j]]]];
        int jdwn = jdwna[j];
        int jdwn2 = jdwna[jdwna[j]];
        int jdwn3 = jdwna[jdwna[jdwna[j]]];
        int jdwn4 = jdwna[jdwna[jdwna[jdwna[j]]]];

        dyphi = (-arr(i,jup4)/280.0  + 4.0*arr(i,jup3)/105.0  - arr(i,jup2)/5.0  + 4.0*arr(i,jup)/5.0  + 
                arr(i,jdwn4)/280.0 - 4.0*arr(i,jdwn3)/105.0 + arr(i,jdwn2)/5.0 - 4.0*arr(i,jdwn)/5.0)/dy;

        return dyphi;
    }


    /* algorithm */
    inline void update_phi(void) {
        int i, j;

        for (i = 0; i < Nx; i++) {
            for (j = 0; j < Ny; j++) {
                phi(i,j) += dt*dphi(i,j);
            }
        }
    }

    inline void calculate_dphi(void) {
        int i, j;
        Field dphidx(Nx,Ny), dphidy(Nx,Ny);
        Field laplacianphi(Nx,Ny);
        Field gradphisq(Nx,Ny); 
        double divJ;

        // dphidx, dphidy
        for (i = 0; i < Nx; i++) {
            for (j = 0; j < Ny; j++) {
                dphidx(i,j) = diff_x(phi,i,j);
                dphidy(i,j) = diff_y(phi,i,j);
            }
        }

        for (i = 0; i < Nx; i++) {
            for (j = 0; j < Ny; j++) {
                // laplacianphi, gradphisq
                laplacianphi(i,j) = laplacian(phi,i,j); 
                gradphisq(i,j) = dphidx(i,j)*dphidx(i,j) + dphidy(i,j)*dphidy(i,j);

                // chemical potential
                mueq(i,j) = -A*phi(i,j) + A*phi(i,j)*phi(i,j)*phi(i,j) - K*laplacianphi(i,j);
                muact(i,j) = lambda*gradphisq(i,j);

                // free energy
                f(i,j) = -0.5*A*phi(i,j)*phi(i,j) + 0.25*A*phi(i,j)*phi(i,j)*phi(i,j)*phi(i,j) + 0.5*K*gradphisq(i,j);

                // noise current
                Lambdax(i,j) = sqrt(2.0*D/(dx*dy*dt))*gaussian_rand();
                Lambday(i,j) = sqrt(2.0*D/(dx*dy*dt))*gaussian_rand();
            }
        }

        // current
        for (i = 0; i < Nx; i++) {
            for (j = 0; j < Ny; j++) {
                // equilibrium currents
                Jxeq(i,j) = -diff_x_8(mueq,i,j); 
                Jyeq(i,j) = -diff_y_8(mueq,i,j);

                // active currents
                Jxact(i,j) = zeta*laplacianphi(i,j)*dphidx(i,j) - diff_x(muact,i,j);
                Jyact(i,j) = zeta*laplacianphi(i,j)*dphidy(i,j) - diff_y(muact,i,j);

                // total current
                Jx(i,j) = Jxeq(i,j) + Jxact(i,j) + Lambdax(i,j);
                Jy(i,j) = Jyeq(i,j) + Jyact(i,j) + Lambday(i,j);
            }
        }

        for (i = 0; i < Nx; i++) {
            for (j = 0; j < Ny; j++) {
                divJ = diff_x_8(Jxeq,i,j) + diff_y_8(Jyeq,i,j) 
                     + diff_x_8(Lambdax,i,j) + diff_y_8(Lambday,i,j)
                     + diff_x(Jxact,i,j) + diff_y(Jyact,i,j);
                dphi(i,j) = -divJ;
            }
        }
    }

private:
    /* initialization */
    inline void initialize(void) {
        int i, j;

        init_math();

        // homogenous phi
        for (i = 0; i < Nx; i++) {
            for (j = 0; j < Ny; j++) {
                phi(i,j) = phi_init;
            }
        }
    }

    // array iup = [1, 2, 3, 4, ..., Nx-1, 0]
    // array iup[iup] = [2, 3, 4, ..., Nx-1, 0, 1]
    inline void init_math(void) {
        int i, j;

        for (i = 0; i < Nx; i++) {
            if (i == Nx - 1) { iupa[i] = 0; } 
            else { iupa[i] = i + 1; }

            if (i == 0) { idwna[i] = Nx - 1; } 
            else { idwna[i] = i - 1; }
        }
        for (j = 0; j < Ny; j++) {
            if (j == Ny - 1) { jupa[j] = 0; } 
            else { jupa[j] = j + 1; }

            if (j == 0) { jdwna[j] = Ny - 1; } 
            else { jdwna[j] = j - 1; }
        }
    }
};
