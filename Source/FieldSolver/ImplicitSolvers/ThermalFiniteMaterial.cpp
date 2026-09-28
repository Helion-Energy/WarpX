/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ThermalFiniteFluxResponse.H"
#include <cctype>
#include <cstdlib>
namespace warpx::thermal::finite_flux {
namespace {
// Temperature degree is counted in half powers. No numerical fitting decides
// the admitted family: the entire owned expression must match this grammar.
struct Degree {int n=0,t=0;Real constant=1.;bool is_constant=true,plain_temperature=false;};
class Reader {
    std::string const& m_text;std::size_t m_at=0;MaterialLaw& m_law;bool m_ok=true;
    void Space(){while(m_at<m_text.size()&&std::isspace(static_cast<unsigned char>(m_text[m_at])))++m_at;}
    bool Take(char c){Space();if(m_at<m_text.size()&&m_text[m_at]==c){++m_at;return true;}return false;}
    bool Token(char const* text,std::size_t size){Space();if(m_text.compare(m_at,size,text)==0){m_at+=size;return true;}return false;}
    void Emit(int op,Real value=0.){if(m_law.size==MaterialLaw::capacity){m_ok=false;return;}int const k=m_law.size++;m_law.operation[k]=op;m_law.constant[k]=value;}
    Real Number(){
        Space();char* end=nullptr;auto const* start=m_text.c_str()+m_at;
        Real const value=std::strtod(start,&end);
        if(end==start||!std::isfinite(value)||value<0.){m_ok=false;return 0.;}
        m_at=static_cast<std::size_t>(end-m_text.c_str());return value;
    }
    Degree Atom(){
        Space();Degree d;
        if(Take('(')){d=Expression();if(!Take(')'))m_ok=false;}
        else if(Token("Te",2)){Emit(MaterialLaw::Temperature);d.t=2;d.is_constant=false;d.plain_temperature=true;}
        else if(Take('n')){Emit(MaterialLaw::Density);d.n=1;d.is_constant=false;}
        else {d.constant=Number();Emit(MaterialLaw::Constant,d.constant);}
        bool const power=Take('^')||Token("**",2);
        if(power){
            Real const exponent=Number();
            if(exponent==2.){
                Emit(MaterialLaw::Square);d.n*=2;d.t*=2;d.constant*=d.constant;
            }else if(exponent==2.5&&d.plain_temperature){
                Emit(MaterialLaw::FiveHalves);d.t=5;
            }else m_ok=false;
            d.plain_temperature=false;
        }
        return d;
    }
    Degree Expression(){
        auto d=Atom();while(m_ok){
            bool const multiply=Take('*');bool const divide=!multiply&&Take('/');if(!multiply&&!divide)break;
            auto const b=Atom();
            if(divide&&(!b.is_constant||!(b.constant>0.))){m_ok=false;break;}
            Emit(multiply?MaterialLaw::Product:MaterialLaw::Quotient);
            d.n+=b.n;d.t+=b.t;d.is_constant=d.is_constant&&b.is_constant;d.plain_temperature=false;
            d.constant=multiply?d.constant*b.constant:d.constant/b.constant;
        }return d;
    }
    static bool Positive(Degree d){return std::isfinite(d.constant)&&d.constant>0.;}
public:
    Reader(std::string const& s,MaterialLaw& law):m_text(s),m_law(law){}
    bool Read(){
        if(Token("min",3)){
            if(!Take('('))return false;
            auto const power=Expression();if(!Take(','))return false;
            auto const cap=Expression();if(!Take(')'))return false;
            if(!Positive(power)||power.n!=0||power.t!=5||!Positive(cap)||cap.n!=1||cap.t!=0)return false;
            Emit(MaterialLaw::Minimum);m_law.family=MaterialLaw::CappedPowerFiveHalves;
        }else{
            auto const d=Expression();m_law.zero=d.is_constant&&d.constant==0.;
            if(m_law.zero)m_law.family=MaterialLaw::Zero;
            else if(Positive(d)&&d.n==1&&d.t==4)m_law.family=MaterialLaw::Quadratic;
            else if(Positive(d)&&d.n==0&&d.t==5)m_law.family=MaterialLaw::PowerFiveHalves;
            else return false;
        }
        Space();return m_ok&&m_at==m_text.size();
    }
};
}
bool ParseMaterial(std::string const& text,MaterialLaw& result){
    MaterialLaw candidate;Reader reader(text,candidate);if(!reader.Read())return false;result=candidate;return true;
}
}
