#include <string>
#include <vector>

#include "loc.h"

namespace dipcc {

const std::vector<std::vector<Loc>> WITH_COASTS{
    {}, // NONE
    {Loc::YOR},
    {Loc::EDI},
    {Loc::LON},
    {Loc::LVP},
    {Loc::NTH},
    {Loc::WAL},
    {Loc::CLY},
    {Loc::NWG},
    {Loc::ENG},
    {Loc::IRI},
    {Loc::NAO},
    {Loc::BEL},
    {Loc::DEN},
    {Loc::HEL},
    {Loc::HOL},
    {Loc::NWY},
    {Loc::SKA},
    {Loc::BAR},
    {Loc::BRE},
    {Loc::MAO},
    {Loc::PIC},
    {Loc::BUR},
    {Loc::RUH},
    {Loc::BAL},
    {Loc::KIE},
    {Loc::SWE},
    {Loc::FIN},
    {Loc::STP, Loc::STP_NC, Loc::STP_SC},
    {Loc::STP, Loc::STP_NC, Loc::STP_SC},
    {Loc::GAS},
    {Loc::PAR},
    {Loc::NAF},
    {Loc::POR},
    {Loc::SPA, Loc::SPA_NC, Loc::SPA_SC},
    {Loc::SPA, Loc::SPA_NC, Loc::SPA_SC},
    {Loc::SPA, Loc::SPA_NC, Loc::SPA_SC},
    {Loc::WES},
    {Loc::MAR},
    {Loc::MUN},
    {Loc::BER},
    {Loc::BOT},
    {Loc::LVN},
    {Loc::PRU},
    {Loc::STP, Loc::STP_NC, Loc::STP_SC},
    {Loc::MOS},
    {Loc::TUN},
    {Loc::LYO},
    {Loc::TYS},
    {Loc::PIE},
    {Loc::BOH},
    {Loc::SIL},
    {Loc::TYR},
    {Loc::WAR},
    {Loc::SEV},
    {Loc::UKR},
    {Loc::ION},
    {Loc::TUS},
    {Loc::NAP},
    {Loc::ROM},
    {Loc::VEN},
    {Loc::GAL},
    {Loc::VIE},
    {Loc::TRI},
    {Loc::ARM},
    {Loc::BLA},
    {Loc::RUM},
    {Loc::ADR},
    {Loc::AEG},
    {Loc::ALB},
    {Loc::APU},
    {Loc::EAS},
    {Loc::GRE},
    {Loc::BUD},
    {Loc::SER},
    {Loc::ANK},
    {Loc::SMY},
    {Loc::SYR},
    {Loc::BUL, Loc::BUL_EC, Loc::BUL_SC},
    {Loc::BUL, Loc::BUL_EC, Loc::BUL_SC},
    {Loc::CON},
    {Loc::BUL, Loc::BUL_EC, Loc::BUL_SC}};

// Return a list of locs including all coastal variants
const std::vector<Loc> &expand_coasts(Loc loc) {
  return WITH_COASTS.at(static_cast<size_t>(loc));
}

const std::vector<Loc> LOCS{
    Loc::YOR,    Loc::EDI,    Loc::LON,   Loc::LVP, Loc::NTH,    Loc::WAL,
    Loc::CLY,    Loc::NWG,    Loc::ENG,   Loc::IRI, Loc::NAO,    Loc::BEL,
    Loc::DEN,    Loc::HEL,    Loc::HOL,   Loc::NWY, Loc::SKA,    Loc::BAR,
    Loc::BRE,    Loc::MAO,    Loc::PIC,   Loc::BUR, Loc::RUH,    Loc::BAL,
    Loc::KIE,    Loc::SWE,    Loc::FIN,   Loc::STP, Loc::STP_NC, Loc::GAS,
    Loc::PAR,    Loc::NAF,    Loc::POR,   Loc::SPA, Loc::SPA_NC, Loc::SPA_SC,
    Loc::WES,    Loc::MAR,    Loc::MUN,   Loc::BER, Loc::BOT,    Loc::LVN,
    Loc::PRU,    Loc::STP_SC, Loc::MOS,   Loc::TUN, Loc::LYO,    Loc::TYS,
    Loc::PIE,    Loc::BOH,    Loc::SIL,   Loc::TYR, Loc::WAR,    Loc::SEV,
    Loc::UKR,    Loc::ION,    Loc::TUS,   Loc::NAP, Loc::ROM,    Loc::VEN,
    Loc::GAL,    Loc::VIE,    Loc::TRI,   Loc::ARM, Loc::BLA,    Loc::RUM,
    Loc::ADR,    Loc::AEG,    Loc::ALB,   Loc::APU, Loc::EAS,    Loc::GRE,
    Loc::BUD,    Loc::SER,    Loc::ANK,   Loc::SMY, Loc::SYR,    Loc::BUL,
    Loc::BUL_EC, Loc::CON,    Loc::BUL_SC};

const std::vector<Loc> ONLY_COAST_LOCS{
    Loc::BUL_EC, Loc::BUL_SC, Loc::SPA_NC,
    Loc::SPA_SC, Loc::STP_NC, Loc::STP_SC,
};

const std::vector<std::string> LOC_STRS{
    "NONE",   "YOR", "EDI",    "LON", "LVP", "NTH", "WAL", "CLY",    "NWG",
    "ENG",    "IRI", "NAO",    "BEL", "DEN", "HEL", "HOL", "NWY",    "SKA",
    "BAR",    "BRE", "MAO",    "PIC", "BUR", "RUH", "BAL", "KIE",    "SWE",
    "FIN",    "STP", "STP/NC", "GAS", "PAR", "NAF", "POR", "SPA",    "SPA/NC",
    "SPA/SC", "WES", "MAR",    "MUN", "BER", "BOT", "LVN", "PRU",    "STP/SC",
    "MOS",    "TUN", "LYO",    "TYS", "PIE", "BOH", "SIL", "TYR",    "WAR",
    "SEV",    "UKR", "ION",    "TUS", "NAP", "ROM", "VEN", "GAL",    "VIE",
    "TRI",    "ARM", "BLA",    "RUM", "ADR", "AEG", "ALB", "APU",    "EAS",
    "GRE",    "BUD", "SER",    "ANK", "SMY", "SYR", "BUL", "BUL/EC", "CON",
    "BUL/SC"};

std::string loc_str(const Loc loc) {
  return LOC_STRS.at(static_cast<size_t>(loc));
}

std::ostream &operator<<(std::ostream &os, Loc loc) {
  return os << loc_str(loc);
}

// >>> [game.map.loc_type.get(loc) == 'WATER' for loc in LOCS]
std::vector<bool> IS_WATER{
    false, false, false, false, true,  false, false, true,  true,  true,  true,
    false, false, true,  false, false, true,  true,  false, true,  false, false,
    false, true,  false, false, false, false, false, false, false, false, false,
    false, false, false, true,  false, false, false, true,  false, false, false,
    false, false, true,  true,  false, false, false, false, false, false, false,
    true,  false, false, false, false, false, false, false, false, true,  false,
    true,  true,  false, false, true,  false, false, false, false, false, false,
    false, false, false, false};

bool is_water(Loc loc) {
  return IS_WATER.at(static_cast<size_t>(loc) - 1); // -1 for NONE
}

// >>> [game.map.area_type(loc) == "COAST" for loc in LOCS]
std::vector<bool> IS_COAST{
    true,  true,  true,  true,  false, true,  true,  false, false, false, false,
    true,  true,  false, true,  true,  false, false, true,  false, true,  false,
    false, false, true,  true,  true,  true,  true,  true,  false, true,  true,
    true,  true,  true,  false, true,  false, true,  false, true,  true,  true,
    false, true,  false, false, true,  false, false, false, false, true,  false,
    false, true,  true,  true,  true,  false, false, true,  true,  false, true,
    false, false, true,  true,  false, true,  false, false, true,  true,  true,
    true,  true,  true,  true};

bool is_coast(Loc loc) {
  return IS_COAST.at(static_cast<size_t>(loc) - 1); // -1 for NONE
}

// >>> [loc[:3] in game.map.scs for loc in LOCS]
// N.B. include center coasts, e.g. IS_CENTER[STP/NC] = true
std::vector<bool> IS_CENTER{
    false, true,  true,  true,  false, false, false, false, false, false, false,
    true,  true,  false, true,  true,  false, false, true,  false, false, false,
    false, false, true,  true,  false, true,  true,  false, true,  false, true,
    true,  true,  true,  false, true,  true,  true,  false, false, false, true,
    true,  true,  false, false, false, false, false, false, true,  true,  false,
    false, false, true,  true,  true,  false, true,  true,  false, false, true,
    false, false, false, false, false, true,  true,  true,  true,  true,  false,
    true,  true,  true,  true};

bool is_center(Loc loc) {
  return IS_CENTER.at(static_cast<size_t>(loc) - 1); // -1 for NONE
}

const std::unordered_map<std::string, Loc> LOC_FROM_STR{
    {"YOR", Loc::YOR},       {"EDI", Loc::EDI},       {"LON", Loc::LON},
    {"LVP", Loc::LVP},       {"NTH", Loc::NTH},       {"WAL", Loc::WAL},
    {"CLY", Loc::CLY},       {"NWG", Loc::NWG},       {"ENG", Loc::ENG},
    {"IRI", Loc::IRI},       {"NAO", Loc::NAO},       {"BEL", Loc::BEL},
    {"DEN", Loc::DEN},       {"HEL", Loc::HEL},       {"HOL", Loc::HOL},
    {"NWY", Loc::NWY},       {"SKA", Loc::SKA},       {"BAR", Loc::BAR},
    {"BRE", Loc::BRE},       {"MAO", Loc::MAO},       {"PIC", Loc::PIC},
    {"BUR", Loc::BUR},       {"RUH", Loc::RUH},       {"BAL", Loc::BAL},
    {"KIE", Loc::KIE},       {"SWE", Loc::SWE},       {"FIN", Loc::FIN},
    {"STP", Loc::STP},       {"STP/NC", Loc::STP_NC}, {"GAS", Loc::GAS},
    {"PAR", Loc::PAR},       {"NAF", Loc::NAF},       {"POR", Loc::POR},
    {"SPA", Loc::SPA},       {"SPA/NC", Loc::SPA_NC}, {"SPA/SC", Loc::SPA_SC},
    {"WES", Loc::WES},       {"MAR", Loc::MAR},       {"MUN", Loc::MUN},
    {"BER", Loc::BER},       {"BOT", Loc::BOT},       {"LVN", Loc::LVN},
    {"PRU", Loc::PRU},       {"STP/SC", Loc::STP_SC}, {"MOS", Loc::MOS},
    {"TUN", Loc::TUN},       {"LYO", Loc::LYO},       {"TYS", Loc::TYS},
    {"PIE", Loc::PIE},       {"BOH", Loc::BOH},       {"SIL", Loc::SIL},
    {"TYR", Loc::TYR},       {"WAR", Loc::WAR},       {"SEV", Loc::SEV},
    {"UKR", Loc::UKR},       {"ION", Loc::ION},       {"TUS", Loc::TUS},
    {"NAP", Loc::NAP},       {"ROM", Loc::ROM},       {"VEN", Loc::VEN},
    {"GAL", Loc::GAL},       {"VIE", Loc::VIE},       {"TRI", Loc::TRI},
    {"ARM", Loc::ARM},       {"BLA", Loc::BLA},       {"RUM", Loc::RUM},
    {"ADR", Loc::ADR},       {"AEG", Loc::AEG},       {"ALB", Loc::ALB},
    {"APU", Loc::APU},       {"EAS", Loc::EAS},       {"GRE", Loc::GRE},
    {"BUD", Loc::BUD},       {"SER", Loc::SER},       {"ANK", Loc::ANK},
    {"SMY", Loc::SMY},       {"SYR", Loc::SYR},       {"BUL", Loc::BUL},
    {"BUL/EC", Loc::BUL_EC}, {"CON", Loc::CON},       {"BUL/SC", Loc::BUL_SC},
};

Loc loc_from_str(const std::string &s) {
  auto it = LOC_FROM_STR.find(s);
  if (it == LOC_FROM_STR.end()) {
    return Loc::NONE;
  } else {
    return it->second;
  }
}

Loc root_loc(Loc loc) {
  switch (loc) {
  case Loc::BUL_EC:
  case Loc::BUL_SC:
    return Loc::BUL;
  case Loc::SPA_NC:
  case Loc::SPA_SC:
    return Loc::SPA;
  case Loc::STP_NC:
  case Loc::STP_SC:
    return Loc::STP;
  default:
    return loc;
  }
}

// >>> '{' + ','.join(['{'+','.join(f'Loc::{c}' for c in game.map.homes[p]) +
// '}' for p in POWERS]) + '}'
// for p in POWERS]) + '}'
const std::vector<std::vector<Loc>> HOME_SCS{
    {Loc::BUD, Loc::TRI, Loc::VIE}, {Loc::EDI, Loc::LON, Loc::LVP},
    {Loc::BRE, Loc::MAR, Loc::PAR}, {Loc::BER, Loc::KIE, Loc::MUN},
    {Loc::NAP, Loc::ROM, Loc::VEN}, {Loc::MOS, Loc::SEV, Loc::STP, Loc::WAR},
    {Loc::ANK, Loc::CON, Loc::SMY}};

const std::vector<std::vector<Loc>> HOME_SCS_ARMY{
    {Loc::BUD, Loc::TRI, Loc::VIE}, {Loc::EDI, Loc::LON, Loc::LVP},
    {Loc::BRE, Loc::MAR, Loc::PAR}, {Loc::BER, Loc::KIE, Loc::MUN},
    {Loc::NAP, Loc::ROM, Loc::VEN}, {Loc::MOS, Loc::SEV, Loc::STP, Loc::WAR},
    {Loc::ANK, Loc::CON, Loc::SMY}};

const std::vector<std::vector<Loc>> HOME_SCS_FLEET{
    {Loc::TRI},
    {Loc::EDI, Loc::LON, Loc::LVP},
    {Loc::BRE, Loc::MAR},
    {Loc::BER, Loc::KIE},
    {Loc::NAP, Loc::ROM, Loc::VEN},
    {Loc::SEV, Loc::STP_NC, Loc::STP_SC},
    {Loc::ANK, Loc::CON, Loc::SMY}};

// Return a vector of root home supply centers for the given power
const std::vector<Loc> &home_centers(Power power) {
  return HOME_SCS.at(static_cast<int>(power) - 1); // -1 for NONE
}

const std::vector<Loc> &home_centers_army(Power power) {
  return HOME_SCS_ARMY.at(static_cast<int>(power) - 1); // -1 for NONE
}

const std::vector<Loc> &home_centers_fleet(Power power) {
  return HOME_SCS_FLEET.at(static_cast<int>(power) - 1); // -1 for NONE
}

} // namespace dipcc
