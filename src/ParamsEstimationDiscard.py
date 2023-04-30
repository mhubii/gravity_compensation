#!/usr/bin/python3
import optas
import sys
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from time import sleep, time, perf_counter, time_ns
from scipy.spatial.transform import Rotation as Rot
from optas.spatialmath import *
import os

import rclpy
import xacro
from ament_index_python import get_package_share_directory
from rclpy import qos
from rclpy.node import Node
import csv

import pathlib

import urdf_parser_py.urdf as urdf

Order = [4,1,2,3,5,6,7]

# def GetOrder(file_path):

#     return order


def RNEA_function(Nb,Nk,rpys,xyzs,axes):
    # 2. RNEA
    Nf = Nb+Nk
    """
    input: q, qdot, qddot, model
    output: tau
    """
    
    om0 = cs.DM([0.0,0.0,0.0])
    om0D = cs.DM([0.0,0.0,0.0])
    gravity_para = cs.DM([0.0, 0.0, -9.81])

    """
    The definination of joint position from joint0 to joint(Nb-1)
    """

    q = cs.SX.sym('q', Nb, 1)
    qd = cs.SX.sym('qd', Nb, 1)
    qdd = cs.SX.sym('qdd', Nb, 1)

    """
    The definination of mass for link1 to linkee
    The definination of center of Mass for link1 to linkee
    The definination of Inertial tensor for link1 to linkee
    """
    m = cs.SX.sym('m', 1, Nb+1)
    cm = cs.SX.sym('cm',3,Nb+1)
    Icm = cs.SX.sym('Icm',3,3*Nb+3)

    """
    external force given by link0
    The list will be appended via RNEA
    """
    fs = [cs.DM([0.0,0.0,0.0])]
    ns = [cs.DM([0.0,0.0,0.0])]


    """
    $$ Forward part of RNEA $$
    oms,omDs,vDs given by link0. The list will be appended via RNEA.
    Notes: the gravity_para represents a base acceration to subsitute the gravity.

    """
    oms = [om0]
    omDs = [om0D]
    vDs = [-gravity_para]
    
    
    # 2.1 forward part of RNEA
    """
    link0->1, link1->2 ...., link7->end-effector (for example)
    joint0,   joint1 ....
    
    """
    
    for i in range(Nf):
        print("link p({0}) to i{1}".format(i,i+1))
        if(i!=Nf-1):
            # print(joints_list_r1)
            iRp = (rpy2r(rpys[i]) @ angvec2r(q[i], axes[i])).T
            iaxisi = iRp @ axes[i]
            omi = iRp @ oms[i] + iaxisi* qd[i]
            omDi = iRp @ omDs[i] +  skew(iRp @oms[i]) @ (iaxisi*qd[i]) + iaxisi*qdd[i]
        else:
            iRp = rpy2r(rpys[i]).T
            omi = iRp @ oms[i]
            omDi = iRp @ omDs[i]

        vDi = iRp @ (vDs[i] 
                        + skew(omDs[i]) @ xyzs[i]
                    + skew(oms[i]) @ (skew(oms[i])@ xyzs[i]))
        
        fi = m[i] * (vDi + skew(omDi)@ cm[:,i]+ skew(omi)@(skew(omi)@cm[:,i]))
        ni = Icm[:,i*3:i*3+3] @ omDi + skew(omi) @ Icm[:,i*3:i*3+3] @ omi #+ skew(cm[:,i]) @ fi
        

        oms.append(omi)
        omDs.append(omDi)
        vDs.append(vDi)
        fs.append(fi)
        ns.append(ni)


    """
    $$ Backward part of RNEA $$
    """

    # pRi = rpy2r(rpys[-1])
    ifi = fs[-1]#cs.DM([0.0,0.0,0.0])
    ini = ns[-1] + skew(cm[:,-1]) @ fs[-1]#cs.DM([0.0,0.0,0.0])
    # ifi = cs.DM([0.0,0.0,0.0])
    # ini = cs.DM([0.0,0.0,0.0])
    taus = []

    # print("Backward: fs[i+1] {0}".format(len(fs)))
    for i in range(Nf-1,0,-1):

        print("Backward: link i({0}) to p{1}".format(i+1,i))
        # print("Backward: fs[i+1]".format(fs[i+1]))
        if(i < Nf-1):
            pRi = rpy2r(rpys[i]) @ angvec2r(q[i], axes[i])
        elif(i == Nf-1):
            pRi = rpy2r(rpys[i])
        else:
            pRi = rpy2r(rpys[i])
        

        ini = ns[i] + pRi @ ini +skew(cm[:,i-1]) @ fs[i] +skew(xyzs[i]) @ pRi @ifi
        ifi= pRi @ ifi + fs[i]
        pRi = rpy2r(rpys[i-1]) @ angvec2r(q[i-1], axes[i-1])
        _tau = ini.T @pRi.T @ axes[i-1]
        taus.append(_tau)


        
    tau_=cs.vertcat(*[taus[k] for k in range(len(taus)-1,-1,-1)])
    dynamics_ = optas.Function('dynamics', [q,qd,qdd,m,cm,Icm], [tau_])
    return dynamics_



def DynamicLinearlization(dynamics_,Nb):
    """
    The definination of joint position from joint0 to joint(Nb-1)
    """

    q = cs.SX.sym('q', Nb, 1)
    qd = cs.SX.sym('qd', Nb, 1)
    qdd = cs.SX.sym('qdd', Nb, 1)

    """
    The definination of mass for link1 to linkee
    The definination of center of Mass for link1 to linkee
    The definination of Inertial tensor for link1 to linkee
    """
    m = cs.SX.sym('m', 1, Nb+1)
    cm = cs.SX.sym('cm',3,Nb+1)
    Icm = cs.SX.sym('Icm',3,3*Nb+3)

    
    Y = []
        
    for i in range(Nb):
        # for every link
        # Y_line = []
        Y_line = []
        # PI_a = []
        for j in range(m.shape[1]):
            # for every parameters
            ## 1. get mass
            m_indu = np.zeros([m.shape[1],m.shape[0]])
            cm_indu = np.zeros([3,Nb+1])#np.zeros([cm.shape[1],cm.shape[0]])
            Icm_indu = np.zeros([3,3*Nb+3])#np.zeros([Icm.shape[1],Icm.shape[0]])
            # print(*m.shape)
            m_indu[j] = 1.0
            # print(m_indu)

            output = dynamics_(q,qd,qdd,m_indu,cm_indu,Icm_indu)[i]
            Y_line.append(output)


            ## 2. get cmx
            output1 = dynamics_(q,qd,qdd,m_indu,cm,Icm_indu)[i]-output
            for k in range(3):
                output_cm = optas.jacobian(output1,cm[k,j])
                output_cm1 = optas.substitute(output_cm,cm,cm_indu)
                Y_line.append(output_cm1)

            ## 3.get Icm
            output2 = dynamics_(q,qd,qdd,m_indu,cm_indu,Icm)[i]-output
            for k in range(3):
                for l in range(k,3,1):
                    output_Icm = optas.jacobian(output2,Icm[k,l+3*j])
                    Y_line.append(output_Icm)

            # sx_lst = optas.horzcat(*Y_seg)
            # Y_line
        sx_lst = optas.horzcat(*Y_line)
        Y.append(sx_lst)
        # print("Y_line shape = {0}, {1}".format(Y_line[0].shape[0],Y_line[0].shape[1]))
        # print("sx_lst shape = {0}, {1}".format(sx_lst.shape[0],sx_lst.shape[1]))

    Y_mat = optas.vertcat(*Y)
    # print(Y_mat)
    print("Y_mat shape = {0}, {1}".format(Y_mat.shape[0],Y_mat.shape[1]))
    Ymat = optas.Function('Dynamic_Ymat',[q,qd,qdd],[Y_mat])

    PI_a = []
    for j in range(m.shape[1]):
        # for every parameters
        pi_temp = [m[j],
                    m[j]*cm[0,j],
                    m[j]*cm[1,j],
                    m[j]*cm[2,j],
                    Icm[0,0+3*j] + m[j]*(cm[1,j]*cm[1,j]+cm[2,j]*cm[2,j]),  # XXi
                    Icm[0,1+3*j] - m[j]*(cm[0,j]*cm[1,j]),  # XYi
                    Icm[0,2+3*j] - m[j]*(cm[0,j]*cm[2,j]),  # XZi
                    Icm[1,1+3*j] + m[j]*(cm[0,j]*cm[0,j]+cm[2,j]*cm[2,j]),  # YYi
                    Icm[1,2+3*j] - m[j]*(cm[1,j]*cm[2,j]),  # YZi
                    Icm[2,2+3*j] + m[j]*(cm[0,j]*cm[0,j]+cm[1,j]*cm[1,j])] # ZZi
        PI_a.append(optas.vertcat(*pi_temp))

    PI_vecter = optas.vertcat(*PI_a)
    print("PI_vecter shape = {0}, {1}".format(PI_vecter.shape[0],PI_vecter.shape[1]))
    PIvector = optas.Function('Dynamic_PIvector',[m,cm,Icm],[PI_vecter])

    return Ymat, PIvector



def find_dyn_parm_deps(dof, parm_num, regressor_func):
    '''
    Find dynamic parameter dependencies (i.e., regressor column dependencies).
    '''

    samples = 10000
    round = 10

    pi = np.pi

    Z = np.zeros((dof * samples, parm_num))

    for i in range(samples):
        q = [float(np.random.random() * 2.0 * pi - pi) for j in range(dof)]
        dq = [float(np.random.random() * 2.0 * pi - pi) for j in range(dof)]
        ddq = [float(np.random.random() * 2.0 * pi - pi)
            for j in range(dof)]
        Z[i * dof: i * dof + dof, :] = np.matrix(
            regressor_func(q, dq, ddq)).reshape(dof, parm_num)

    R1_diag = np.linalg.qr(Z, mode='r').diagonal().round(round)
    dbi = []
    ddi = []
    for i, e in enumerate(R1_diag):
        if e != 0:
            dbi.append(i)
        else:
            ddi.append(i)
    dbn = len(dbi)

    P = np.mat(np.eye(parm_num))[:, dbi + ddi]
    Pb = P[:, :dbn]
    Pd = P[:, dbn:]

    Rbd1 = np.mat(np.linalg.qr(Z * P, mode='r'))
    Rb1 = Rbd1[:dbn, :dbn]
    Rd1 = Rbd1[:dbn, dbn:]

    Kd = np.mat((np.linalg.inv(Rb1) * Rd1).round(round))

    return Pb, Pd, Kd



class Estimator(Node):
    def __init__(self, node_name = "para_estimatior", dt_ = 5.0, N_ = 100) -> None:
        super().__init__(node_name=node_name)

        self.dt_ = dt_
        self.declare_parameter("model", "med7")
        self.model_ = str(self.get_parameter("model").value)
        path = os.path.join(
            get_package_share_directory("lbr_description"),
            "urdf",
            self.model_,
            f"{self.model_}.urdf.xacro",
        )
        self.N = N_

        # self.lbr_command_timer_ = self.create_timer(self.dt_, self.timer_cb_regressor)

        # 1. Get the kinematic parameters of every joints
        self.robot = optas.RobotModel(
            xacro_filename=path,
            time_derivs=[1],  # i.e. joint velocity
        )
        root = self.robot.urdf.get_root()
        ee_link = "lbr_link_ee"
        xyzs, rpys, axes = [], [], []


        joints_list = self.robot.urdf.get_chain(root, ee_link, links=False)
        print("joints_list = {0}"
              .format(joints_list)
              )
        # assumption: The first joint is fixed. The information in this joint is not recorded
        """
        xyzs starts from joint 0 to joint ee
        rpys starts from joint 0 to joint ee
        axes starts from joint 0 to joint ee
        """
        joints_list_r = joints_list[1:]
        for joint_name in joints_list_r:
            print(joint_name)
            joint = self.robot.urdf.joint_map[joint_name]
            xyz, rpy = self.robot.get_joint_origin(joint)
            axis = self.robot.get_joint_axis(joint)

            # record the kinematic parameters
            xyzs.append(xyz)
            rpys.append(rpy)
            axes.append(axis)
        print("xyz, rpy, axis = {0}, {1} ,{2}".format(xyzs, rpys, axes))

        # 2. RNEA
        """
        input: q, qdot, qddot, model
        output: tau
        """
        Nb = self.robot.ndof
        self.Nb = Nb
        om0 = cs.DM([0.0,0.0,0.0])
        om0D = cs.DM([0.0,0.0,0.0])
        gravity_para = cs.DM([0.0, 0.0, -9.81])

        """
        The definination of joint position from joint0 to joint(Nb-1)
        """

        q = cs.SX.sym('q', Nb, 1)
        qd = cs.SX.sym('qd', Nb, 1)
        qdd = cs.SX.sym('qdd', Nb, 1)

        """
        The definination of mass for link1 to linkee
        The definination of center of Mass for link1 to linkee
        The definination of Inertial tensor for link1 to linkee
        """
        m = cs.SX.sym('m', 1, Nb+1)
        cm = cs.SX.sym('cm',3,Nb+1)
        Icm = cs.SX.sym('Icm',3,3*Nb+3)

        """
        external force given by link0
        The list will be appended via RNEA
        """
        fs = [cs.DM([0.0,0.0,0.0])]
        ns = [cs.DM([0.0,0.0,0.0])]


        """
        $$ Forward part of RNEA $$
        oms,omDs,vDs given by link0. The list will be appended via RNEA.
        Notes: the gravity_para represents a base acceration to subsitute the gravity.

        """
        oms = [om0]
        omDs = [om0D]
        vDs = [-gravity_para]
        
        
        # 2.1 forward part of RNEA
        """
        link0->1, link1->2 ...., link7->end-effector (for example)
        joint0,   joint1 ....
        
        """
        joints_list_r1 = joints_list_r
        for i in range(len(joints_list_r1)):
            print("link p({0}) to i{1}".format(i,i+1))
            if(i!=len(joints_list_r1)-1):
                # print(joints_list_r1)
                iRp = (rpy2r(rpys[i]) @ angvec2r(q[i], axes[i])).T
                iaxisi = iRp @ axes[i]
                omi = iRp @ oms[i] + iaxisi* qd[i]
                omDi = iRp @ omDs[i] +  skew(iRp @oms[i]) @ (iaxisi*qd[i]) + iaxisi*qdd[i]
            else:
                iRp = rpy2r(rpys[i]).T
                omi = iRp @ oms[i]
                omDi = iRp @ omDs[i]

            vDi = iRp @ (vDs[i] 
                         + skew(omDs[i]) @ xyzs[i]
                        + skew(oms[i]) @ (skew(oms[i])@ xyzs[i]))
            
            fi = m[i] * (vDi + skew(omDi)@ cm[:,i]+ skew(omi)@(skew(omi)@cm[:,i]))
            ni = Icm[:,i*3:i*3+3] @ omDi + skew(omi) @ Icm[:,i*3:i*3+3] @ omi #+ skew(cm[:,i]) @ fi
            

            oms.append(omi)
            omDs.append(omDi)
            vDs.append(vDi)
            fs.append(fi)
            ns.append(ni)


        """
        $$ Backward part of RNEA $$
        """

        # pRi = rpy2r(rpys[-1])
        ifi = fs[-1]#cs.DM([0.0,0.0,0.0])
        ini = ns[-1] + skew(cm[:,-1]) @ fs[-1]#cs.DM([0.0,0.0,0.0])
        # ifi = cs.DM([0.0,0.0,0.0])
        # ini = cs.DM([0.0,0.0,0.0])
        taus = []

        # print("Backward: fs[i+1] {0}".format(len(fs)))
        for i in range(len(joints_list_r)-1,0,-1):

            print("Backward: link i({0}) to p{1}".format(i+1,i))
            # print("Backward: fs[i+1]".format(fs[i+1]))
            if(i < len(joints_list_r)-1):
                pRi = rpy2r(rpys[i]) @ angvec2r(q[i], axes[i])
            elif(i == len(joints_list_r)-1):
                pRi = rpy2r(rpys[i])
            else:
                pRi = rpy2r(rpys[i])
            

            ini = ns[i] + pRi @ ini +skew(cm[:,i-1]) @ fs[i] +skew(xyzs[i]) @ pRi @ifi
            ifi= pRi @ ifi + fs[i]
            pRi = rpy2r(rpys[i-1]) @ angvec2r(q[i-1], axes[i-1])
            _tau = ini.T @pRi.T @ axes[i-1]
            taus.append(_tau)


        
        tau_=cs.vertcat(*[taus[k] for k in range(len(taus)-1,-1,-1)])
        self.dynamics_ = optas.Function('dynamics', [q,qd,qdd,m,cm,Icm], [tau_])
        print(tau_.size())
        urdf_string_ = xacro.process(path)
        robot = urdf.URDF.from_xml_string(urdf_string_)
        print([joint.name for joint in robot.joints if joint.origin is not None])
        print([joint.origin.xyz for joint in robot.joints if joint.origin is not None])

        print([link.name for link in robot.links if link.inertial is not None])
        print([link.inertial.origin.xyz for link in robot.links if link.inertial is not None])
        print([link.inertial.mass for link in robot.links if link.inertial is not None])
        masses = [link.inertial.mass for link in robot.links if link.inertial is not None]#+[1.0]
        self.masses_np = np.array(masses[1:])
        print("masses = {0}".format(self.masses_np))

        massesCenter = [link.inertial.origin.xyz for link in robot.links if link.inertial is not None]#+[[0.0,0.0,0.0]]
        self.massesCenter_np = np.array(massesCenter[1:]).T
        # Inertia = [np.mat(link.inertial.inertia.to_matrix()) for link in robot.links if link.inertial is not None]
        Inertia = [link.inertial.inertia.to_matrix() for link in robot.links if link.inertial is not None]

        self.Inertia_np = np.hstack(tuple(Inertia[1:]))
        # print("massesCenter = {0}".format(self.massesCenter_np))
        # print("Inertia = {0}".format(self.Inertia_np))
        # print("Inertia = {0} , {1},{2}".format(np.size(self.Inertia_np,0),np.size(self.Inertia_np,1),np.size(self.Inertia_np,2)))

        # tau_ = tau
        
        g_ =  self.dynamics_(q,np.zeros([Nb,1]),np.zeros([Nb,1]),m,cm,Icm)
        g1 =  self.dynamics_(q,np.zeros([Nb,1]),np.zeros([Nb,1]),m,cm,np.zeros([3,3*Nb+3]))
        g1_ = optas.simplify(g1)
        self.gra = optas.Function('gravity', [q,m,cm], [g1_])
        # print("g1_ = {0}".format(g1_[0]))

        """
        Get Inertia parameters set
        """
        # dynamicsF_ = self.dynamics_(q,np.zeros([Nb,1]),np.zeros([Nb,1]),
        #                             np.ones([Nb+1,1]),np.zeros([3,Nb+1]),np.zeros([3,3*Nb+3]))


        # print("dynamicsF_ = {0}".format(dynamicsF_))


        
        # _numK = 10
        # for every torque
        Y = []
        
        for i in range(tau_.shape[0]):
            # for every link
            # Y_line = []
            Y_line = []
            # PI_a = []
            for j in range(m.shape[1]):
                # for every parameters
                ## 1. get mass
                m_indu = np.zeros([m.shape[1],m.shape[0]])
                cm_indu = np.zeros([3,Nb+1])#np.zeros([cm.shape[1],cm.shape[0]])
                Icm_indu = np.zeros([3,3*Nb+3])#np.zeros([Icm.shape[1],Icm.shape[0]])
                # print(*m.shape)
                m_indu[j] = 1.0
                # print(m_indu)

                output = self.dynamics_(q,qd,qdd,m_indu,cm_indu,Icm_indu)[i]
                Y_line.append(output)


                ## 2. get cmx
                output1 = self.dynamics_(q,qd,qdd,m_indu,cm,Icm_indu)[i]-output
                for k in range(3):
                    output_cm = optas.jacobian(output1,cm[k,j])
                    output_cm1 = optas.substitute(output_cm,cm,cm_indu)
                    Y_line.append(output_cm1)

                ## 3.get Icm
                output2 = self.dynamics_(q,qd,qdd,m_indu,cm_indu,Icm)[i]-output
                for k in range(3):
                    for l in range(k,3,1):
                        output_Icm = optas.jacobian(output2,Icm[k,l+3*j])
                        Y_line.append(output_Icm)

                # sx_lst = optas.horzcat(*Y_seg)
                # Y_line
            sx_lst = optas.horzcat(*Y_line)
            Y.append(sx_lst)
            # print("Y_line shape = {0}, {1}".format(Y_line[0].shape[0],Y_line[0].shape[1]))
            # print("sx_lst shape = {0}, {1}".format(sx_lst.shape[0],sx_lst.shape[1]))

        Y_mat = optas.vertcat(*Y)
        # print(Y_mat)
        print("Y_mat shape = {0}, {1}".format(Y_mat.shape[0],Y_mat.shape[1]))
        self.Ymat = optas.Function('Dynamic_Ymat',[q,qd,qdd],[Y_mat])

        PI_a = []
        for j in range(m.shape[1]):
            # for every parameters
            pi_temp = [m[j],
                        m[j]*cm[0,j],
                        m[j]*cm[1,j],
                        m[j]*cm[2,j],
                        Icm[0,0+3*j] + m[j]*(cm[1,j]*cm[1,j]+cm[2,j]*cm[2,j]),  # XXi
                        Icm[0,1+3*j] - m[j]*(cm[0,j]*cm[1,j]),  # XYi
                        Icm[0,2+3*j] - m[j]*(cm[0,j]*cm[2,j]),  # XZi
                        Icm[1,1+3*j] + m[j]*(cm[0,j]*cm[0,j]+cm[2,j]*cm[2,j]),  # YYi
                        Icm[1,2+3*j] - m[j]*(cm[1,j]*cm[2,j]),  # YZi
                        Icm[2,2+3*j] + m[j]*(cm[0,j]*cm[0,j]+cm[1,j]*cm[1,j])] # ZZi
            PI_a.append(optas.vertcat(*pi_temp))

        PI_vecter = optas.vertcat(*PI_a)
        print("PI_vecter shape = {0}, {1}".format(PI_vecter.shape[0],PI_vecter.shape[1]))
        self.PIvector = optas.Function('Dynamic_PIvector',[m,cm,Icm],[PI_vecter])
    
    @ staticmethod
    def readCsvToList(path):
        l = []
        with open(path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                joint_names = [x.strip() for x in list(row.keys())]
                l.append([float(x) for x in row.values()])
        return l    
    

    def ExtractFromCsv(self):
        path_pos = os.path.join(
            get_package_share_directory("gravity_compensation"),
            "test",
            "joint_states_positions.csv",
        )

        path_vel = os.path.join(
            get_package_share_directory("gravity_compensation"),
            "test",
            "joint_states_velocities.csv",
        )

        path_eff = os.path.join(
            get_package_share_directory("gravity_compensation"),
            "test",
            "joint_states_efforts.csv",
        )
        pos = Estimator.readCsvToList(path_pos)
        vel = Estimator.readCsvToList(path_vel)
        eff = Estimator.readCsvToList(path_eff)

        # print("pos = {0}".format(pos))
        
        return pos, vel, eff

    

    def timer_cb_regressor(self, positions, velocities, efforts):
        
        Pb, Pd, Kd =find_dyn_parm_deps(7,80,self.Ymat)
        K = Pb.T +Kd @Pd.T

        q_nps = []
        qd_nps = []
        qdd_nps = []
        taus = []
        Y_ = []
        init_para = np.random.uniform(0.0, 0.1, size=50)
        
        for k in range(300,len(positions),1):
            # print("q_np = {0}".format(q_np))
            # q_np = np.random.uniform(-1.5, 1.5, size=7)
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]

            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np)-np.array(qdlast_np))/(velocities[k][0]-velocities[k-1][0])
            qdd_np = qdd_np.tolist()
            
            # qd_np = np.random.uniform(-0.2, 0.2, size=7)
            # qdd_np = np.random.uniform(-0.1, 0.1, size=7)

            # tau_ext = self.robot.rnea(q_np,qd_np,qdd_np)
            

            Y_temp = self.Ymat(q_np,
                               qd_np,
                               qdd_np) @Pb 
            Y_.append(Y_temp)
            # print(Y_temp)
            

            q_nps.append(q_np)
            qd_nps.append(qd_np)
            qdd_nps.append(qdd_np)

            taus.append(tau_ext)
            # print(qdd_np)

        
        Y_r = optas.vertcat(*Y_)
        q_nps1 = np.hstack(q_nps)
        qd_nps1 = np.hstack(qd_nps)
        qdd_nps1 = np.hstack(qdd_nps)
        taus1 = np.hstack(taus)

        # solution=self.regress(init_para, q_nps1,qd_nps1,qdd_nps1,taus1)
        # print("solution = {0}".format(solution[f"{self.pam_name}/y"]))

        print("taus1 size = {0}".format(taus1.shape))
        print("q_nps1 size = {0}".format(q_nps1.shape))
        print("qd_nps1 size = {0}".format(qd_nps1.shape))

        # real_pam=self.PIvector(self.masses_np,self.massesCenter_np,self.Inertia_np)
        # print("error = {0}".format(solution[f"{self.pam_name}/y"] - real_pam))
        # q_np = np.random.uniform(-1.5, 1.5, size=7)
        # qd_np = np.random.uniform(-1.5, 1.5, size=7)
        # qdd_np = np.random.uniform(-1.5, 1.5, size=7)
        # print(Y_r)
        taus1 = taus1.T

        #temp =Y_r.T @ Y_r 
        #qdd_np
        estimate_pam = np.linalg.inv(Y_r.T @ Y_r) @ Y_r.T @ taus1

        return estimate_pam
    
    def testWithEstimatedPara(self, positions, velocities, efforts, para)->None:

        Pb, Pd, Kd =find_dyn_parm_deps(7,80,self.Ymat)
        K = Pb.T +Kd @Pd.T

        for k in range(1,len(positions),1):
            # q_np = positions[k][4,1,2,3,5,6,7]
            # qd_np = velocities[k][4,1,2,3,5,6,7]
            # tau_ext = efforts[k][4,1,2,3,5,6,7]
            # qdd_np = (np.array(velocities[k][4,1,2,3,5,6,7])-np.array(velocities[k-1][4,1,2,3,5,6,7]))/(velocities[k][0]-velocities[k-1][0])
            # qdd_np = qdd_np.tolist()

            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]

            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np)-np.array(qdlast_np))/(velocities[k][0]-velocities[k-1][0])
            qdd_np = qdd_np.tolist()

            # tau_ext = self.robot.rnea(q_np,qd_np,qdd_np)
            # e=self.Ymat(q_np,qd_np,qdd_np)@Pb @ (solution[f"{self.pam_name}/y"] -  K @real_pam)
            # print("error = {0}".format(e))

            e=self.Ymat(q_np,qd_np,qdd_np)@Pb @  para - tau_ext 
            print("error1 = {0}".format(e))
            print("tau_ext = {0}".format(tau_ext))

        # print("taus1 size = {0}".format(taus1.shape))
        # print("q_nps1 size = {0}".format(q_nps1.shape))
        # print("qd_nps1 size = {0}".format(qd_nps1.shape))

        # real_pam=self.PIvector(self.masses_np,self.massesCenter_np,self.Inertia_np)







def main(args=None):
    rclpy.init(args=args)
    paraEstimator = Estimator()
    positions, velocities, efforts = paraEstimator.ExtractFromCsv()
    estimate_pam = paraEstimator.timer_cb_regressor(positions, velocities, efforts)
    print("estimate_pam = {0}".format(estimate_pam))
    paraEstimator.testWithEstimatedPara(positions, velocities, efforts,estimate_pam)

    rclpy.shutdown()



if __name__ == "__main__":
    main()