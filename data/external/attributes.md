## Structure des attributs de sortie (`out.*`)

Les attributs de sortie sont organisés en **9 groupes fonctionnels** croisant
la fonction d'usage du ménage et le type d'énergie consommée.

---

### Groupe 1 — Confort thermique (chauffage & climatisation)

| Attribut | Unité | Description |
|---|---|---|
| `out.electricity.heating.energy_consumption..kwh` | kWh | Électricité consommée par le chauffage (hors ventilateurs/pompes) |
| `out.electricity.heating_fans_pumps.energy_consumption..kwh` | kWh | Électricité des ventilateurs/pompes de distribution de chaleur |
| `out.electricity.heating_hp_bkup.energy_consumption..kwh` | kWh | Électricité de l'appoint de la pompe à chaleur |
| `out.electricity.heating_hp_bkup_fa.energy_consumption..kwh` | kWh | Électricité des ventilateurs/pompes pendant l'appoint PAC |
| `out.electricity.cooling.energy_consumption..kwh` | kWh | Électricité consommée par la climatisation (hors ventilateurs/pompes) |
| `out.electricity.cooling_fans_pumps.energy_consumption..kwh` | kWh | Électricité des ventilateurs/pompes de distribution de froid |
| `out.natural_gas.heating.energy_consumption..kwh` | kWh | Gaz naturel consommé par le chauffage (hors appoint PAC) |
| `out.fuel_oil.heating.energy_consumption..kwh` | kWh | Fioul consommé par le chauffage (hors appoint PAC) |
| `out.propane.heating.energy_consumption..kwh` | kWh | Propane consommé par le chauffage (hors appoint PAC) |
| `out.load.heating.energy_delivered..kbtu` | kBtu | Énergie thermique livrée par le chauffage (pertes de distribution incluses) |
| `out.load.cooling.energy_delivered..kbtu` | kBtu | Énergie thermique livrée par la climatisation (pertes de distribution incluses) |
| `out.load.heating.peak..kbtu_per_hr` | kBtu/hr | Pic de charge chauffage sur 60 min |
| `out.load.cooling.peak..kbtu_per_hr` | kBtu/hr | Pic de charge refroidissement |
| `out.capacity.heating..btu_per_hr` | BTU/hr | Capacité installée du système de chauffage |
| `out.capacity.cooling..btu_per_hr` | BTU/hr | Capacité installée du système de climatisation |
| `out.capacity.heat_pump_backup..btu_per_hr` | BTU/hr | Capacité de l'appoint de la pompe à chaleur |
| `out.unmet_hours.heating..hr` | hr | Heures où le setpoint de chauffage n'est pas atteint |
| `out.unmet_hours.cooling..hr` | hr | Heures où le setpoint de climatisation n'est pas atteint |
| `out.params.size_heating_system_primary..kbtu_per_hr` | kBtu/hr | Taille du système de chauffage principal |
| `out.params.size_heating_system_secondary..kbtu_per_hr` | kBtu/hr | Taille du système de chauffage secondaire |
| `out.params.size_cooling_system_primary..kbtu_per_hr` | kBtu/hr | Taille du système de climatisation principal |
| `out.params.size_heat_pump_backup_primary..kbtu_per_hr` | kBtu/hr | Taille de l'appoint de la pompe à chaleur |
| `out.params.duct_unconditioned_surface_area..ft2` | ft² | Surface des gaines en espace non conditionné |
| `out.params.hvac_geothermal_loop_borehole_trench_count` | — | Nombre de forages géothermiques |
| `out.params.hvac_geothermal_loop_borehole_trench_length..ft` | ft | Longueur d'un forage géothermique |

---

### Groupe 2 — Eau chaude sanitaire

| Attribut | Unité | Description |
|---|---|---|
| `out.electricity.hot_water.energy_consumption..kwh` | kWh | Électricité consommée par le chauffe-eau (hors pompe de recirculation) |
| `out.electricity.hot_water_solar_th.energy_consumption..kwh` | kWh | Électricité du système solaire thermique eau chaude |
| `out.natural_gas.hot_water.energy_consumption..kwh` | kWh | Gaz naturel consommé par le chauffe-eau |
| `out.fuel_oil.hot_water.energy_consumption..kwh` | kWh | Fioul consommé par le chauffe-eau |
| `out.propane.hot_water.energy_consumption..kwh` | kWh | Propane consommé par le chauffe-eau |
| `out.load.hot_water.energy_delivered..kbtu` | kBtu | Énergie livrée par le système ECS (contributions solaires incluses) |
| `out.load.hot_water.energy_solar_thermal..kbtu` | kBtu | Énergie livrée par le chauffe-eau solaire thermique |
| `out.load.hot_water.energy_tank_losses..kbtu` | kBtu | Pertes thermiques du ballon d'eau chaude |
| `out.params.size_water_heater..gal` | gal | Capacité du ballon d'eau chaude (gallons) |

---

### Groupe 3 — Hygiène & nettoyage

| Attribut | Unité | Description |
|---|---|---|
| `out.electricity.clothes_washer.energy_consumption..kwh` | kWh | Électricité consommée par le lave-linge |
| `out.electricity.clothes_dryer.energy_consumption..kwh` | kWh | Électricité consommée par le sèche-linge |
| `out.natural_gas.clothes_dryer.energy_consumption..kwh` | kWh | Gaz naturel consommé par le sèche-linge |
| `out.propane.clothes_dryer.energy_consumption..kwh` | kWh | Propane consommé par le sèche-linge |
| `out.electricity.dishwasher.energy_consumption..kwh` | kWh | Électricité consommée par le lave-vaisselle |
| `out.hot_water.clothes_washer..gal` | gal | Eau chaude consommée par le lave-linge |
| `out.hot_water.dishwasher..gal` | gal | Eau chaude consommée par le lave-vaisselle |
| `out.hot_water.fixtures..gal` | gal | Eau chaude consommée par douches, lavabos et bains |
| `out.hot_water.distribution_waste..gal` | gal | Eau chaude perdue dans les canalisations (eau froide en attente) |

---

### Groupe 4 — Alimentation (cuisiner & conserver)

| Attribut | Unité | Description |
|---|---|---|
| `out.electricity.range_oven.energy_consumption..kwh` | kWh | Électricité consommée par la cuisinière / le four |
| `out.natural_gas.range_oven.energy_consumption..kwh` | kWh | Gaz naturel consommé par la cuisinière / le four |
| `out.propane.range_oven.energy_consumption..kwh` | kWh | Propane consommé par la cuisinière / le four |
| `out.electricity.refrigerator.energy_consumption..kwh` | kWh | Électricité consommée par le réfrigérateur principal |
| `out.electricity.freezer.energy_consumption..kwh` | kWh | Électricité consommée par le congélateur indépendant |
| `out.natural_gas.grill.energy_consumption..kwh` | kWh | Gaz naturel consommé par le barbecue gaz extérieur |

---

### Groupe 5 — Divertissement & loisirs

| Attribut | Unité | Description |
|---|---|---|
| `out.electricity.television.energy_consumption..kwh` | kWh | Électricité consommée par la télévision |
| `out.electricity.plug_loads.energy_consumption..kwh` | kWh | Électricité des prises (PC, consoles, chargeurs…) |
| `out.electricity.ceiling_fan.energy_consumption..kwh` | kWh | Électricité consommée par les ventilateurs de plafond |
| `out.electricity.pool_pump.energy_consumption..kwh` | kWh | Électricité consommée par la pompe de piscine |
| `out.electricity.pool_heater.energy_consumption..kwh` | kWh | Électricité consommée par le chauffe-piscine |
| `out.natural_gas.pool_heater.energy_consumption..kwh` | kWh | Gaz naturel consommé par le chauffe-piscine |
| `out.electricity.permanent_spa_pump.energy_consumption..kwh` | kWh | Électricité consommée par la pompe de spa |
| `out.electricity.permanent_spa_heat.energy_consumption..kwh` | kWh | Électricité consommée par le chauffe-spa |
| `out.natural_gas.permanent_spa_heat.energy_consumption..kwh` | kWh | Gaz naturel consommé par le chauffe-spa |

---

### Groupe 6 — Éclairage & ambiance

| Attribut | Unité | Description |
|---|---|---|
| `out.electricity.lighting_interior.energy_consumption..kwh` | kWh | Électricité consommée par l'éclairage intérieur |
| `out.electricity.lighting_exterior.energy_consumption..kwh` | kWh | Électricité consommée par l'éclairage extérieur (déco de Noël incluse) |
| `out.electricity.lighting_garage.energy_consumption..kwh` | kWh | Électricité consommée par l'éclairage du garage |
| `out.natural_gas.lighting.energy_consumption..kwh` | kWh | Gaz naturel consommé par l'éclairage extérieur (lanternes) |
| `out.natural_gas.fireplace.energy_consumption..kwh` | kWh | Gaz naturel consommé par la cheminée décorative |

---

### Groupe 7 — Mobilité (véhicule électrique)

| Attribut | Unité | Description |
|---|---|---|
| `out.electricity.ev_charging.energy_consumption..kwh` | kWh | Électricité consommée pour la recharge du VE à domicile |
| `out.unmet_hours.ev_driving..hr` | hr | Heures où le VE ne peut pas démarrer (batterie insuffisante) |

---

### Groupe 8 — Infrastructure & services du logement

| Attribut | Unité | Description |
|---|---|---|
| `out.electricity.mech_vent.energy_consumption..kwh` | kWh | Électricité consommée par la VMC (hors préchauffage/prérefroidissement) |
| `out.params.flow_rate_mechanical_ventilation..cfm` | CFM | Débit d'air de la ventilation mécanique |
| `out.electricity.well_pump.energy_consumption..kwh` | kWh | Électricité consommée par la pompe de puits |

---

### Groupe 9 — Production & bilan énergétique global

| Attribut | Unité | Description |
|---|---|---|
| `out.electricity.pv.energy_consumption..kwh` | kWh | Énergie produite par les panneaux PV (valeur négative = production) |
| `out.electricity.net.energy_consumption..kwh` | kWh | Consommation électrique nette (après déduction de la production PV) |
| `out.electricity.total.energy_consumption..kwh` | kWh | Consommation électrique brute (battery charging/discharging inclus) |
| `out.natural_gas.total.energy_consumption..kwh` | kWh | Consommation totale de gaz naturel |
| `out.fuel_oil.total.energy_consumption..kwh` | kWh | Consommation totale de fioul (toutes variantes incluses) |
| `out.propane.total.energy_consumption..kwh` | kWh | Consommation totale de propane |
| `out.site_energy.net.energy_consumption..kwh` | kWh | Bilan énergétique total toutes sources (PV déduit) |
| `out.site_energy.total.energy_consumption..kwh` | kWh | Énergie totale du site (battery incluse) |
| `out.qoi.electricity.maximum_daily_peak_summer..kw` | kW | Pic de demande électrique journalier — été (juin/juil/août) |
| `out.qoi.electricity.maximum_daily_peak_winter..kw` | kW | Pic de demande électrique journalier — hiver (déc/jan/fév) |
| `out.utility_bills.electricity_bill..usd` | USD | Facture annuelle électricité |
| `out.utility_bills.natural_gas_bill..usd` | USD | Facture annuelle gaz naturel |
| `out.utility_bills.fuel_oil_bill..usd` | USD | Facture annuelle fioul |
| `out.utility_bills.propane_bill..usd` | USD | Facture annuelle propane |
| `out.utility_bills.total_bill..usd` | USD | Facture annuelle totale toutes énergies |
| `out.energy_burden..percentage` | % | Part du revenu du ménage consacrée à l'énergie |
| `out.emissions.total.lrmer_mid_case_25..co2e_kg` | kg CO₂e | Émissions totales — scénario de référence LRMER Mid Case 25 ans |
| `out.emissions.electricity.lrmer_mid_case_25..co2e_kg` | kg CO₂e | Émissions électricité — scénario LRMER Mid Case 25 ans |
| `out.emissions.fuel_oil.total..co2e_kg` | kg CO₂e | Émissions liées à la combustion de fioul sur site |
| `out.emissions.natural_gas.total..co2e_kg` | kg CO₂e | Émissions liées à la combustion de gaz naturel sur site |
| `out.emissions.propane.total..co2e_kg` | kg CO₂e | Émissions liées à la combustion de propane sur site |

---

### Paramètres physiques du bâtiment (`out.params.*`)

> Ces attributs décrivent les caractéristiques physiques du logement,
> pas des consommations énergétiques.

| Attribut | Unité | Description |
|---|---|---|
| `out.params.door_area..ft2` | ft² | Surface des portes extérieures |
| `out.params.floor_area_attic..ft2` | ft² | Surface du plancher des combles |
| `out.params.floor_area_attic_insulation_increase..ft2_delta_r_value` | R-value | Surface combles × augmentation d'isolation |
| `out.params.floor_area_conditioned_infiltration_reduction..ft2_delta_ach50` | ACH50 | Surface conditionnée × réduction d'infiltration |
| `out.params.floor_area_foundation..ft2` | ft² | Surface du plancher des fondations |
| `out.params.floor_area_lighting..ft2` | ft² | Surface utilisée pour le calcul de l'éclairage |
| `out.params.rim_joist_area_above_grade_exterior..ft2` | ft² | Surface des solives de rive au-dessus du sol |
| `out.params.roof_area..ft2` | ft² | Surface de toiture |
| `out.params.slab_perimeter_exposed_conditioned..ft` | ft | Périmètre de dalle exposé (espace conditionné) |
| `out.params.wall_area_above_grade_conditioned..ft2` | ft² | Surface des murs au-dessus du sol (espace conditionné) |
| `out.params.wall_area_above_grade_exterior..ft2` | ft² | Surface des murs au-dessus du sol (extérieur) |
| `out.params.wall_area_below_grade..ft2` | ft² | Surface des murs enterrés |
| `out.params.window_area..ft2` | ft² | Surface des fenêtres |

---

