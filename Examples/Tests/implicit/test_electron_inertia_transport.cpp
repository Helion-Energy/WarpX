/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>

namespace
{
    using namespace amrex::literals;
    using Field = ablastr::fields::VectorField;
    using warpx::fields::FieldType;
    using amrex::MultiFab;

    struct Scratch
    {
        std::array<MultiFab,3> values;
        Field field;
        explicit Scratch (Field const& like)
        {
            for (int c = 0; c < 3; ++c) {
                values[c].define(like[c]->boxArray(),like[c]->DistributionMap(),1,
                                 like[c]->nGrowVect());
                field[c] = &values[c];
            }
        }
        void copy (Field const& source)
        {
            for (int c = 0; c < 3; ++c) {
                MultiFab::Copy(values[c],*source[c],0,0,1,values[c].nGrowVect());
            }
        }
    };

    amrex::Real product (Field const& a, Field const& b, amrex::Geometry const& geom)
    {
        amrex::Real result = 0.0_rt;
        for (int c = 0; c < 3; ++c) {
            MultiFab measured(a[c]->boxArray(),a[c]->DistributionMap(),1,0);
            auto const volume = MakeQdsmcVolumeElement(geom,a[c]->ixType());
            for (amrex::MFIter mfi(measured); mfi.isValid(); ++mfi) {
                auto const aa = a[c]->const_array(mfi), bb = b[c]->const_array(mfi);
                auto const out = measured.array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    out(i,j,k) = volume(i,j,k)*aa(i,j,k)*bb(i,j,k);
                });
            }
            result += measured.sum_unique(0,false,geom.periodicity());
        }
        return result;
    }

    amrex::Real electron_energy (WarpX& w, HybridPICModel& hp, MultiFab const& rho,
                                 Field const& Ji, Field const& B)
    {
        hp.CalculatePlasmaCurrent(B,w.GetEBUpdateEFlag()[0],0);
        auto const J = w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
        auto const& geom = w.Geom(0);
        auto const* pedestal = hp.DensityPedestal(0);
        amrex::GpuArray<int,3> const node{1,1,1}, ratio{1,1,1};
        auto const floor = hp.m_n_floor*PhysConst::q_e;
        amrex::Real result = 0.0_rt;
        for (int c = 0; c < 3; ++c) {
            auto const it = Ji[c]->ixType().toIntVect();
            amrex::GpuArray<int,3> const stagger{it[0],it[1],1};
            auto const volume = MakeQdsmcVolumeElement(geom,Ji[c]->ixType());
            MultiFab measured(Ji[c]->boxArray(),Ji[c]->DistributionMap(),1,0);
            for (amrex::MFIter mfi(measured); mfi.isValid(); ++mfi) {
                auto const den = rho.const_array(mfi);
                auto const ped = pedestal ? pedestal->const_array(mfi)
                                         : amrex::Array4<amrex::Real const>{};
                auto const ji = Ji[c]->const_array(mfi), jtot = J[c]->const_array(mfi);
                auto const out = measured.array(mfi);
                bool const has_pedestal = pedestal != nullptr;
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    auto const raw = ablastr::coarsen::sample::Interp(
                        den,node,stagger,ratio,i,j,k,0) + (has_pedestal ?
                        ablastr::coarsen::sample::Interp(ped,node,stagger,ratio,i,j,k,0) : 0.0_rt);
                    auto const q = ji(i,j,k)-jtot(i,j,k);
                    out(i,j,k) = volume(i,j,k)*PhysConst::m_e*q*q/
                        (2.0_rt*PhysConst::q_e*amrex::max(floor,raw));
                });
            }
            result += measured.sum_unique(0,false,geom.periodicity());
        }
        return result;
    }

    // Full represented thermal state, including the pedestal and floor.
    // QDSMCClassEnergy deliberately omits sub-floor nodes for its bulk/band
    // report, so it is not the total electron energy of a moving pedestal edge.
    amrex::Real thermal_energy (WarpX& w, HybridPICModel const& hp, MultiFab const& rho)
    {
        if (!hp.m_solve_electron_energy_equation) { return 0.0_rt; }
        auto const& te=*w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
        auto const* pedestal=hp.DensityPedestal(0);
        bool const has_ped=pedestal!=nullptr;
        auto const floor=hp.m_n_floor*PhysConst::q_e;
        auto const volume=MakeQdsmcVolumeElement(w.Geom(0),rho.ixType());
        auto const capacity=PhysConst::kb/(PhysConst::q_e*(hp.m_gamma-1.0_rt));
        MultiFab measured(rho.boxArray(),rho.DistributionMap(),1,0);
        for (amrex::MFIter mfi(measured);mfi.isValid();++mfi) {
            auto const den=rho.const_array(mfi), temp=te.const_array(mfi);
            auto const ped=has_ped ? pedestal->const_array(mfi) : amrex::Array4<amrex::Real const>{};
            auto const out=measured.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                out(i,j,k)=volume(i,j,k)*capacity*temp(i,j,k)*
                    amrex::max(floor,den(i,j,k)+(has_ped ? ped(i,j,k) : 0.0_rt));
            });
        }
        return measured.sum_unique(0,false,w.Geom(0).periodicity());
    }

    std::array<amrex::Real,2> charge_radius (MultiFab const& rho, amrex::Geometry const& geom,
                                           bool const corrected_axis)
    {
        MultiFab measured(rho.boxArray(),rho.DistributionMap(),2,0);
        auto const dx = geom.CellSizeArray();
        auto const volume = MakeQdsmcVolumeElement(geom,rho.ixType());
        for (amrex::MFIter mfi(measured); mfi.isValid(); ++mfi) {
            auto const den = rho.const_array(mfi);
            auto const out = measured.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                auto const weight = volume(i,j,k)*(corrected_axis && i == 0 ? 4.0_rt/3.0_rt : 1.0_rt);
                out(i,j,k,0) = weight*den(i,j,k);
                out(i,j,k,1) = weight*den(i,j,k)*(i*dx[0])*(i*dx[0]);
            });
        }
        auto const charge = measured.sum_unique(0,false,geom.periodicity());
        return {charge,std::sqrt(measured.sum_unique(1,false,geom.periodicity())/charge)};
    }

    // The split driver subtracts w(rho_i)*E_ext in Ohm's law and adds
    // E_ext back afterwards. For this uniform-B drive, curl(curl(E_ext))=0,
    // so its residual external electron force is (1-w)*E_ext. Its work
    // q_electron_velocity.(1-w)*E_ext belongs in addition to wall Poynting.
    amrex::Real penetration_power (WarpX& w, HybridPICModel const& hp,
                                   MultiFab const& rho, Field const& Ji, Field const& Eext)
    {
        auto const J=w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
        auto const& geom=w.Geom(0);
        auto const floor=hp.m_n_floor*PhysConst::q_e;
        auto const smooth=hp.m_n_floor_smooth_width*floor;
        amrex::GpuArray<int,3> const node{1,1,1},ratio{1,1,1};
        amrex::Real result=0.0_rt;
        for (int c=0;c<3;++c) {
            MultiFab measured(Ji[c]->boxArray(),Ji[c]->DistributionMap(),1,0);
            auto const it=Ji[c]->ixType().toIntVect();
            amrex::GpuArray<int,3> const st{it[0],it[1],1};
            auto const volume=MakeQdsmcVolumeElement(geom,Ji[c]->ixType());
            for (amrex::MFIter mfi(measured);mfi.isValid();++mfi) {
                auto const ji=Ji[c]->const_array(mfi),j=J[c]->const_array(mfi);
                auto const ex=Eext[c]->const_array(mfi),den=rho.const_array(mfi);
                auto const out=measured.array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j0,int k) {
                    auto const raw=ablastr::coarsen::sample::Interp(den,node,st,ratio,i,j0,k,0);
                    auto const weight=HybridExtSubWeight(raw,floor,smooth);
                    out(i,j0,k)=volume(i,j0,k)*(ji(i,j0,k)-j(i,j0,k))*(1.0_rt-weight)*ex(i,j0,k);
                });
            }
            result+=measured.sum_unique(0,false,geom.periodicity());
        }
        return result;
    }

    // Measure the actual applied native motional field. Its numerical work
    // is diagnostic only; it is never booked as physical heat or external input.
    amrex::Real hall_work (WarpX& w, HybridPICModel& hp, MultiFab const& rho,
                          Field const& Ji, Field const& B)
    {
        auto J=w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
        Scratch force(Ji), q(Ji);
        MultiFab pressure(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
        pressure.setVal(0.0_rt);
        bool const old_pressure=hp.m_include_electron_pressure_term;
        bool const old_external=hp.m_add_external_fields;
        hp.m_include_electron_pressure_term=false;
        hp.m_add_external_fields=false;
        for (int c=0;c<3;++c) { force.values[c].setVal(0.0_rt); }
        w.get_pointer_fdtd_solver_fp(0)->HybridPICSolveE(
            force.field,J,Ji,B,rho,pressure,w.GetEBUpdateEFlag()[0],0,&hp,false,false);
        hp.m_include_electron_pressure_term=old_pressure;
        hp.m_add_external_fields=old_external;
        q.copy(Ji);
        for (int c=0;c<3;++c) { MultiFab::Subtract(q.values[c],*J[c],0,0,1,0); }
        return product(q.field,force.field,w.Geom(0));
    }

    amrex::Real wall_power (Field const& E, Field const& B, amrex::Geometry const& geom)
    {
        MultiFab measured(E[1]->boxArray(),E[1]->DistributionMap(),1,0);
        auto const nr = geom.Domain().bigEnd(0)+1;
        auto const dz = geom.CellSize(1), radius = geom.ProbHi(0);
        for (amrex::MFIter mfi(measured); mfi.isValid(); ++mfi) {
            auto const et = E[1]->const_array(mfi), ez = E[2]->const_array(mfi);
            auto const bt = B[1]->const_array(mfi), bz = B[2]->const_array(mfi);
            auto const out = measured.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                out(i,j,k) = 0.0_rt;
                if (i == nr) {
                    // Discrete Yee Poynting flux paired with the magnetic
                    // dual-volume sum: use the last interior tangential B.
                    // The two products keep their native z staggering; their
                    // periodic sums share an index but are not averaged first.
                    out(i,j,k) = 2.0_rt*MathConst::pi*radius*dz/PhysConst::mu0*
                        (et(i,j,k)*bz(i-1,j,k)-ez(i,j,k)*bt(i-1,j,k));
                }
            });
        }
        return measured.sum_unique(0,false,geom.periodicity());
    }
}

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& hp = *w.get_pointer_HybridPICModel();
        auto& particles = w.GetPartContainer();
        auto const& geom = w.Geom(0);
        auto& rho = *w.m_fields.get(FieldType::rho_fp,0);
        auto const Ji = w.m_fields.get_alldirs(FieldType::current_fp,0);
        auto const B = w.m_fields.get_alldirs(FieldType::Bfield_fp,0);
        auto const E = w.m_fields.get_alldirs(FieldType::Efield_fp,0);
        int steps = 100;
        bool prescribed = true, assert_energy = true, corrected_axis = true, magnetic_piston = false;
        amrex::ParmParse("boundary").query("verboncoeur_axis_correction",corrected_axis);
        amrex::Real tolerance = 0.01_rt, minimum_crossing = 2.0_rt;
        amrex::ParmParse test("edge_transport");
        test.query("steps",steps); test.query("prescribed",prescribed);
        test.query("assert_energy",assert_energy);
        test.query("magnetic_piston",magnetic_piston);
        test.query("energy_relative_tolerance",tolerance);
        test.query("minimum_crossed_cells",minimum_crossing);
        AMREX_ALWAYS_ASSERT(geom.isPeriodic(1) && hp.m_include_electron_inertia_elliptic);
        if (prescribed) {
            AMREX_ALWAYS_ASSERT(!hp.m_add_external_fields);
        }
        if (magnetic_piston) {
            AMREX_ALWAYS_ASSERT(!hp.m_add_external_fields && !prescribed);
            auto const dx=geom.CellSizeArray();
            auto const radius=0.5_rt*geom.ProbHi(0),width=0.1_rt*geom.ProbHi(0);
            for (int c=0;c<3;++c) {
                auto const it=B[c]->ixType().toIntVect();
                for (amrex::MFIter mfi(*B[c]);mfi.isValid();++mfi) {
                    auto const b=B[c]->array(mfi);
                    amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        auto const r=(i+(it[0]?0.0_rt:0.5_rt))*dx[0];
                        b(i,j,k)=c==2 ? 0.1_rt+0.05_rt*(1.0_rt+
                            std::tanh((r*r-radius*radius)/(2.0_rt*radius*width))) : 0.0_rt;
                    });
                }
                B[c]->FillBoundary(geom.periodicity());
            }
            w.ApplyBfieldBoundary(0,PatchType::fine,SubcyclingHalf::None,0.0_rt);
        }
        w.HybridPICInitializeRhoJandB();
        Scratch previous_current(Ji), centered(Ji), previous_b(B), previous_e(E), previous_ext(E);
        if (hp.m_add_external_fields) {
            AMREX_ALWAYS_ASSERT(!hp.m_external_unified && !hp.m_external_e_subtraction_unconditional);
        }
        MultiFab previous_rho(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
        MultiFab continuity(rho.boxArray(),rho.DistributionMap(),1,0);
        auto& pressure=*w.m_fields.get(FieldType::hybrid_electron_pressure_fp,0);
        MultiFab previous_pressure(pressure.boxArray(),pressure.DistributionMap(),1,pressure.nGrowVect());
        MultiFab evolved_pressure(pressure.boxArray(),pressure.DistributionMap(),1,pressure.nGrowVect());
        amrex::Real const dt = w.getdt(0);
        amrex::Real initial_energy = 0.0_rt, initial_radius = 0.0_rt, initial_charge = 0.0_rt;
        amrex::Real initial_electron=0.0_rt,initial_ion=0.0_rt,initial_thermal=0.0_rt,initial_magnetic=0.0_rt;
        amrex::Real work = 0.0_rt, previous_power = 0.0_rt, defect = 0.0_rt;
        amrex::Real hall_energy=0.0_rt,previous_hall=0.0_rt;
        amrex::Real final_crossing = 0.0_rt, max_continuity = 0.0_rt;
        amrex::Real max_crossing = 0.0_rt, max_defect = 0.0_rt;
        for (int step = 0; step <= steps; ++step) {
            MultiFab::Copy(previous_rho,rho,0,0,1,rho.nGrowVect());
            previous_current.copy(Ji); previous_b.copy(B); previous_e.copy(E);
            if (prescribed && hp.m_solve_electron_energy_equation) {
                MultiFab::Copy(previous_pressure,pressure,0,0,1,pressure.nGrowVect());
            }
            if (hp.m_add_external_fields) {
                previous_ext.copy(w.m_fields.get_alldirs(FieldType::hybrid_E_fp_external,0));
            }
            auto const ion = particles.GetParticleContainerFromName("ions").sumParticleEnergy();
            auto const thermal = thermal_energy(w,hp,rho);
            auto const magnetic = product(B,B,geom)/(2.0_rt*PhysConst::mu0);
            auto const moment = charge_radius(rho,geom,corrected_axis);
            if (prescribed) {
                particles.PushX(dt); particles.Redistribute();
                w.HybridPICEvolveFields();
            } else {
                w.Evolve(1);
            }
            w.ComputeRZContinuityResidual(continuity,previous_rho,rho,Ji,dt);
            auto const cont = continuity.norminf()*dt/std::max(rho.norminf(),previous_rho.norminf());
            max_continuity = std::max(max_continuity,cont);
            for (int c = 0; c < 3; ++c) {
                MultiFab::LinComb(centered.values[c],0.5_rt,previous_current.values[c],0,
                    0.5_rt,*Ji[c],0,0,1,Ji[c]->nGrowVect());
            }
            auto const electron = electron_energy(w,hp,previous_rho,centered.field,previous_b.field);
            amrex::Real power, penetration=0.0_rt, poynting=0.0_rt;
            if (prescribed) {
                // Reevaluate the same production first-half Ohm RHS for the
                // measured t^n current. Ions follow prescribed ballistic paths;
                // -Ji.E is the external constraint work entering electron energy.
                if (hp.m_solve_electron_energy_equation) {
                    MultiFab::Copy(evolved_pressure,pressure,0,0,1,pressure.nGrowVect());
                    MultiFab::Copy(pressure,previous_pressure,0,0,1,pressure.nGrowVect());
                }
                hp.HybridPICSolveE(E,centered.field,previous_b.field,previous_rho,
                                  w.GetEBUpdateEFlag()[0],0,false);
                power = -product(centered.field,E,geom);
                if (hp.m_solve_electron_energy_equation) {
                    MultiFab::Copy(pressure,evolved_pressure,0,0,1,pressure.nGrowVect());
                }
            } else {
                poynting = -wall_power(previous_e.field,previous_b.field,geom);
                if (hp.m_add_external_fields) {
                    penetration=penetration_power(w,hp,previous_rho,centered.field,previous_ext.field);
                }
                power = poynting+penetration;
            }
            auto const numerical_hall=prescribed?0.0_rt:hall_work(w,hp,previous_rho,centered.field,previous_b.field);
            auto const total = prescribed ? electron+thermal+magnetic : electron+ion+thermal+magnetic;
            if (step == 0) {
                initial_energy = total; initial_radius = moment[1]; initial_charge = moment[0];
                initial_electron=electron; initial_ion=ion; initial_thermal=thermal; initial_magnetic=magnetic;
            } else {
                work += 0.5_rt*dt*(power+previous_power);
                hall_energy += 0.5_rt*dt*(numerical_hall+previous_hall);
            }
            // In a closed compression the net energy change should vanish;
            // normalize by energy transferred between reservoirs, not by that
            // vanishing net change. Externally driven/prescribed tests retain
            // their imposed-work normalization.
            auto const exchange = prescribed ? 0.0_rt : 0.5_rt*(std::abs(electron-initial_electron)+
                std::abs(ion-initial_ion)+std::abs(thermal-initial_thermal)+std::abs(magnetic-initial_magnetic));
            auto const scale = std::max({std::abs(work),std::abs(total-initial_energy),exchange,
                                        0.01_rt*std::abs(initial_energy),1.e-30_rt});
            defect = std::abs(total-initial_energy-work)/scale;
            final_crossing = (initial_radius-moment[1])/geom.CellSize(0);
            max_crossing = std::max(max_crossing,final_crossing);
            max_defect = std::max(max_defect,defect);
            amrex::Real emax = 0.0_rt;
            for (int c = 0; c < 3; ++c) {
                AMREX_ALWAYS_ASSERT(E[c]->is_finite() && B[c]->is_finite() && Ji[c]->is_finite());
                emax = std::max(emax,E[c]->norminf());
            }
            AMREX_ALWAYS_ASSERT(rho.is_finite() && std::isfinite(total));
            auto log = amrex::Print();
            log << std::setprecision(17) << "EDGE_TRANSPORT {\"step\":" << step
                << ",\"time_s\":" << step*dt << ",\"prescribed\":" << prescribed
                << ",\"electron_bulk_J\":" << electron << ",\"ion_J\":" << ion
                << ",\"thermal_J\":" << thermal << ",\"magnetic_J\":" << magnetic
                << ",\"wall_power_W\":" << poynting << ",\"penetration_power_W\":" << penetration
                << ",\"numerical_hall_power_W\":" << numerical_hall
                << ",\"numerical_hall_work_J\":" << hall_energy
                << ",\"power_W\":" << power << ",\"work_J\":" << work
                << ",\"energy_scale_J\":" << scale << ",\"energy_relative_defect\":" << defect
                << ",\"continuity_relative\":" << cont
                << ",\"charge_relative\":" << (moment[0]-initial_charge)/initial_charge
                << ",\"radius_rms_m\":" << moment[1]
                << ",\"crossed_cells\":" << final_crossing << ",\"E_max_V_m\":" << emax
                << ",\"rho_max_C_m3\":" << previous_rho.norminf() << "}\n";
            previous_power = power; previous_hall = numerical_hall;
            hp.CalculatePlasmaCurrent(B,w.GetEBUpdateEFlag()[0],0);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_continuity < 2.e-10_rt,
            "Deposited moving edge must obey the filtered ion continuity relation");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_crossing >= minimum_crossing,
            "The edge must cross multiple radial cells");
        if (assert_energy) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_defect < tolerance,
                "The moving-edge inertia work must close its energy budget");
        }
        amrex::Print() << "EDGE_TRANSPORT_COMPLETE\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
