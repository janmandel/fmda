
import numpy as np


class GridMoistureModel:

    Tk = np.array([1.0, 10.0, 100.0, 1000.0]) * 3600    # nominal fuel delays
    r0 = 0.05                                           # threshold rainfall [mm/h]
    rk = 8.0                                            # saturation rain intensity [mm/h]
    Trk = 14.0 * 3600                                   # time constant for wetting model [s]
    S = 2.5                                             # saturation intensity [dimensionless]


    def __init__(self, m0, Tk = None, P0 = None):
        """
        Initialize the model with given position and moisture levels.

          m0  - the initial condition for the entire grid (shape: grid0 x grid1 x num_fuels)
          Tk  - drying/wetting time constants of simulated fuels (one per fuel), default [1 10 100 1000]
          P0  - initial state error covariance

        """
        s0,s1,k = m0.shape
        if Tk is not None:
            self.Tk = Tk
        dim = k + 2
        assert k == len(self.Tk)
        self.m_ext = np.zeros((s0,s1,dim))
        self.m_ext[:,:,:k] = m0

        # note: the moisture advance will proceed by fuel moisture types
        # thus we only need space for one class at a time
        self.m_i = np.zeros((s0,s1))
        #self.mn_i = np.zeros((s0,s1))
        self.rlag = np.zeros((s0,s1))
        self.equi = np.zeros((s0,s1))
        self.model_ids = np.zeros((s0,s1))

        self.EdA = np.zeros((s0,s1))
        self.EwA = np.zeros((s0,s1))

        # state covariance current and forecasted
        self.P = np.zeros((s0,s1,dim,dim))
        self.P2 = np.zeros((dim,dim))
        for s in np.ndindex((s0,s1)):
          self.P[s[0],s[1],:,:] = P0

        # fill out the fixed parts of the jacobian
        self.J = np.zeros((s0,s1,dim,dim))
        self.Jii = np.zeros((s0,s1))

        # note: the observation operator H is common for all dims
        self.H = np.zeros((k, dim))


    def advance_model(self, Ed, Ew, r, dt, mQ = None):
        """
        This model captures the moisture dynamics at each grid point independently.

        Ed - drying equilibrium [grid shape]
        Ew - wetting equilibrium [grid shape]
        r - rain intensity for time unit [grid shape] [mm/h]
        dt - integration step [scalar] [s]
        mQ - the proess noise matrix [shape dim x dim, common across grid points]
        """
        Tk = self.Tk
        k = Tk.shape[0]                 # number of fuel classes
        m_ext = self.m_ext
        equi = self.equi
        rlag = self.rlag
        model_ids = self.model_ids
        m_i = self.m_i
        #mn_i = self.mn_i
        J = self.J
        Jii = self.Jii
        P2 = self.P2
        EdA, EwA = self.EdA, self.EwA

        # add assimilated equilibrium difference, which is shared across spatial locations
        EdA[:,:] = Ed
        EdA += m_ext[:,:,k]
        EwA[:,:] = Ew
        EwA += m_ext[:,:,k]

        dS = m_ext[:,:,k+1]

#        print('grid: Ed = %g Ew = %g r = %g dlt_E = %g dlt_S = %g EdA = %g EwA = %g S = %g rk = %g r0 = %g' %
#              (Ed[0,0], Ew[0,0], r[0,0], m_ext[0,0,k], m_ext[0,0,k+1], EdA[0,0], EwA[0,0], self.S, self.rk, self.r0))

        assert np.all(EdA >= EwA)
        
        # re-initialize the Jacobian (in case we must recompute it)
        J[:] = 0.0
        J[:,:,k,k] = 1.0
        J[:,:,k+1,k+1] = 1.0

        # where rainfall is above threshold (spatially different), apply
        # saturation model, equi and rlag are specific to fuel type and
        # location
        for i in range(k):

            # initialize equilibrium with current moisture
            m_i[:,:] = m_ext[:,:,i]
            equi[:,:] = m_i
            rlag[:,:] = 0.0
            model_ids[:] = 4

            # on grid locations where there is rain, modify equilibrium
            has_rain = r > self.r0
            no_rain = np.logical_not(has_rain)

            equi[has_rain] = self.S + dS[has_rain]
            rlag[has_rain] = 1.0 / self.Trk * (1.0 - np.exp(- (r[has_rain] - self.r0) / self.rk))
            model_ids[has_rain] = 3

            # equilibrium is selected according to current moisture level
            is_drying = np.logical_and(no_rain, m_i > EdA)
            equi[is_drying] = EdA[is_drying]
            model_ids[is_drying] = 1

            is_wetting = np.logical_and(no_rain, m_i < EwA)
            equi[is_wetting] = EwA[is_wetting]
            model_ids[is_wetting] = 2

            rlag[no_rain] = 1.0 / Tk[i]

            dead_zone = (model_ids  == 4)

#            print('grid model_ids = %d EwA = %g EdA = %g m[i] = %g' % (model_ids[0,0], EwA[0,0], EdA[0,0],m_i[0,0]))

            # select appropriate integration method according to change for each fuel 
            # and location
            rlag *= dt
            change = rlag
            big_change = change > 0.01
            m_i[big_change] += (equi[big_change] - m_i[big_change]) * (1.0 - np.exp(-change[big_change]))
            small_change = np.logical_not(big_change)
            m_i[small_change] += (equi[small_change] - m_i[small_change]) * change[small_change] * (1.0 - 0.5 * change[small_change])

            # store in state matrix
            m_ext[:,:,i] = m_i


            # update model state covariance if requested using the old state (the jacobian must be computed as well)
            if mQ is not None:

               # partial m_i/partial m_i
                Jii[big_change] = np.exp(-change[big_change])
                Jii[small_change] = 1.0 - change[small_change] * (1.0 - 0.5 * change[small_change])
                Jii[dead_zone] = 1.0
                J[:,:,i,i] = Jii

                # partial E/partial m_i
                Jii[:] = 0.0
                norain_big = np.logical_and(no_rain, big_change)
                norain_small = np.logical_and(no_rain, small_change)
                Jii[norain_big] = 1.0 - np.exp(-change[norain_big])
                Jii[norain_small] = change[norain_small] * (1.0 - 0.5 * change[norain_small])
                Jii[dead_zone] = 0.0
                J[:,:,i,k] = Jii

                # partial S/partial m_i
                Jii[:] = 0.0
                rain_big = np.logical_and(has_rain, big_change)
                rain_small = np.logical_and(has_rain, small_change)
                Jii[rain_big] = 1.0 - np.exp(-change[rain_big])
                Jii[rain_small] = change[rain_small] * (1.0 - 0.5 * change[rain_small])
                Jii[dead_zone] = 0.0
                J[:,:,i,k+1] = Jii


        # transformed to run in-place with one pre-allocated temporary
        for i,j in np.ndindex(self.P[:,:,0,0].shape):
            # Ps = Js P[s,:,:] Js^T
            Ps = self.P[i,j,:,:]
            Js = J[i,j,:,:]
            np.dot(Js, Ps, P2)
            np.dot(P2, Js.T, Ps)

            # P[s,:,:] = Js P[s,:,:] Js^T + mQ
            Ps += mQ
            self.P[i,j,:,:] = Ps


    def get_state(self):
        """
        Return the current state for all grid points. READ-ONLY under normal circumstances.
        """
        return self.m_ext


    def get_state_covar(self):
        """
        Return the state covariance for all grid points. READ-ONLY under normal circumstances.
        """
        return self.P


    def kalman_update(self, O, V, fuel_types, Kg = None):
        """
        Updates the state using the observation at the grid point.

          O - the observations [shape grid_size x len(fuel_types)]
          V - the (diagonal) variance matrix of the measurements [shape grid_size x len(fuel_types) x len(fuel_types)]
          fuel_types - the fuel types for which the observations exist (used to construct observation vector)

        """
        m_ext, P, H, P2 = self.m_ext, self.P, self.H, self.P2
        H[:] = 0.0

        # construct an observation operator tailored to observed fuel types 
        for i in fuel_types:
            H[i,i] = 1.0
        Ho = H[fuel_types,:]

        for i,j in np.ndindex(P[:,:,0,0].shape):
            # re-use P2 to store forecast covariance for position s
            P2[:,:] = P[i,j,:,:]

            if np.any(np.diag(P2) < 0.0):
              print("ERROR: negative diagonal %s" % (str(np.diag(P2))))
              aaa

            # compute Kalman gain
            I = np.dot(np.dot(Ho, P2), Ho.T) + V[i,j,:,:]
            K = np.dot(np.dot(P2, Ho.T), np.linalg.inv(I))

            if K[1,0] > 1.0 or K[1,0] < 0.0:
              print("ERROR at %s, Kalman gain %g out of bounds! I=%g V=%g" % (str((i,j)), K[fuel_type,0], I, V[s[0],s[1],0,0]))

            # update state and state covariance
            m_ext[i,j,:] += np.dot(K, O[i,j,:] - m_ext[i,j,fuel_types])
            P[i,j,:,:] -= np.dot(np.dot(K, Ho), P2)

            if Kg is not None:
                Kg[i,j,:] = K[:,0]


    def kalman_update_single(self, O, V, fuel_type, Kg = None):
        """
        Perform an optimized Kalman update assuming there is only one fuel type.
        """
        m_ext, P, H, P2 = self.m_ext, self.P, self.H, self.P2

        #print("******* GRID MODEL ********")
        #print("grid: O = %.18e  V = %.18e   fuel_type = %d" % (O[0,0], V[0,0,0,0], fuel_type))

        for i,j in np.ndindex(P[:,:,0,0].shape):
            P2[:,:] = P[i,j,:,:]
            #print(P2)

            if np.any(np.diag(P2) < 0.0):
                print("ERROR: negative diagonal %s" % (str(np.diag(P2))))
                aaa

            #print("grid: P2[:,ft] = %.18e %.18e %.18e %.18e %.18e %.18e" %
            #    (P2[0,1], P2[1,1], P2[2,1], P2[3,1], P2[4,1], P2[5,1]))

            I = P2[fuel_type,fuel_type] + V[i,j,0,0]
            K = P2[:,fuel_type] / I

            #print("grid: inv(I) = %.18e" % (1.0/I))

            #print("grid: P2[ft,ft] = %.18e I = %.18e K = %.18e %.18e %.18e %.18e %.18e %.18e" %
            #    (P2[1,1], I, K[0], K[1], K[2], K[3], K[4], K[5]))

            if K[fuel_type] > 1.0 or K[fuel_type] < 0.0:
              print("ERROR at %s, Kalman gain %g out of bounds! I=%g V=%g" % (str((i,j)), K[fuel_type], I, V[i,j,0,0]))

            m_ext[i,j,:] += K * (O[i,j,0] - m_ext[i,j,fuel_type])

            #print(np.outer(K,P2[fuel_type,:]))

            P2 -= np.dot(K[:,np.newaxis], P2[fuel_type:fuel_type+1,:])
            P[i,j,:,:] = P2

           # print("grid: P2[ft,ft] = %.18e" % P2[1,1])

            if Kg is not None:
                Kg[i,j,:] = K


