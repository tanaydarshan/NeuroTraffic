package neurotraffic

import org.apache.spark.sql.{SparkSession, SaveMode}
import scala.util.Random

/**
 * Generates realistic mock data so you can test the full GraphX pipeline
 * without waiting for Navaneeth's cleaning pipeline.
 *
 * Run with:
 *   spark-submit --class neurotraffic.SampleDataGenerator neurotraffic-graphx.jar [local|hdfs]
 *
 * "local" writes to ./sample_data/ (for testing on your laptop)
 * "hdfs"  writes to hdfs:///data/     (for testing on the cluster)
 */
object SampleDataGenerator {

  val rand = new Random(42) // fixed seed for reproducible data

  // ----- Real NYC taxi zone data (subset of 30 key zones) -----
  // These are actual zone IDs and names from the TLC dataset.
  val taxiZones: Seq[(Int, String, Double, Double, String)] = Seq(
    (4,   "Alphabet City",           40.7258, -73.9815, "Manhattan"),
    (12,  "Battery Park City",       40.7117, -74.0154, "Manhattan"),
    (13,  "Bedford-Stuyvesant",      40.6872, -73.9418, "Brooklyn"),
    (24,  "Borough Park",            40.6345, -73.9927, "Brooklyn"),
    (43,  "Central Park",            40.7829, -73.9654, "Manhattan"),
    (48,  "Clinton East",            40.7614, -73.9923, "Manhattan"),
    (50,  "Clinton Hill",            40.6889, -73.9662, "Brooklyn"),
    (68,  "East Chelsea",            40.7448, -73.9955, "Manhattan"),
    (79,  "East Village",            40.7265, -73.9815, "Manhattan"),
    (87,  "Financial District North",40.7088, -74.0094, "Manhattan"),
    (88,  "Financial District South",40.7033, -74.0131, "Manhattan"),
    (90,  "Flatiron",                40.7401, -73.9903, "Manhattan"),
    (100, "Garment District",        40.7536, -73.9918, "Manhattan"),
    (107, "Gramercy",                40.7382, -73.9860, "Manhattan"),
    (113, "Greenwich Village North", 40.7336, -73.9991, "Manhattan"),
    (114, "Greenwich Village South", 40.7299, -74.0005, "Manhattan"),
    (125, "Hudson Square",           40.7268, -74.0073, "Manhattan"),
    (137, "Kips Bay",                40.7424, -73.9800, "Manhattan"),
    (140, "Lenox Hill East",         40.7650, -73.9590, "Manhattan"),
    (141, "Lenox Hill West",         40.7695, -73.9625, "Manhattan"),
    (142, "Lincoln Square East",     40.7731, -73.9836, "Manhattan"),
    (143, "Lincoln Square West",     40.7751, -73.9862, "Manhattan"),
    (148, "Lower East Side",         40.7150, -73.9843, "Manhattan"),
    (158, "Meatpacking/West Village",40.7390, -74.0058, "Manhattan"),
    (161, "Midtown Center",          40.7549, -73.9840, "Manhattan"),
    (162, "Midtown East",            40.7551, -73.9712, "Manhattan"),
    (163, "Midtown North",           40.7625, -73.9779, "Manhattan"),
    (164, "Midtown South",           40.7506, -73.9879, "Manhattan"),
    (166, "Morningside Heights",     40.8097, -73.9621, "Manhattan"),
    (170, "Murray Hill",             40.7489, -73.9769, "Manhattan"),
    (186, "Penn Station/Madison Sq W",40.7502,-73.9930, "Manhattan"),
    (202, "Prospect Heights",        40.6775, -73.9692, "Brooklyn"),
    (209, "Randalls Island",         40.7932, -73.9212, "Manhattan"),
    (224, "SoHo",                    40.7233, -73.9985, "Manhattan"),
    (229, "Stuyvesant Heights",      40.6836, -73.9400, "Brooklyn"),
    (230, "Stuyvesant Town",         40.7318, -73.9785, "Manhattan"),
    (231, "Sunset Park East",        40.6465, -74.0069, "Brooklyn"),
    (234, "Times Square/Theatre District",40.7580,-73.9855,"Manhattan"),
    (236, "Upper East Side North",   40.7736, -73.9566, "Manhattan"),
    (237, "Upper East Side South",   40.7669, -73.9588, "Manhattan"),
    (239, "Upper West Side North",   40.7990, -73.9680, "Manhattan"),
    (243, "Upper West Side South",   40.7831, -73.9741, "Manhattan"),
    (246, "West Chelsea/Hudson Yards",40.7504,-74.0020, "Manhattan"),
    (249, "West Village",            40.7336, -74.0027, "Manhattan"),
    (261, "Williamsburg (North)",    40.7145, -73.9565, "Brooklyn"),
    (262, "Williamsburg (South)",    40.7081, -73.9571, "Brooklyn"),
    (263, "Yorkville East",          40.7767, -73.9483, "Manhattan")
  )

  // ----- Real NYC subway stations (subset of 40 key stations) -----
  val subwayStations: Seq[(Int, String, Double, Double, String, String)] = Seq(
    (1,  "Times Sq-42 St",        40.7559, -73.9871, "Manhattan", "1237NQRSW"),
    (2,  "Grand Central-42 St",   40.7527, -73.9772, "Manhattan", "4567S"),
    (3,  "34 St-Herald Sq",       40.7498, -73.9877, "Manhattan", "BDFMNQRW"),
    (4,  "14 St-Union Sq",        40.7356, -73.9910, "Manhattan", "456LNQRW"),
    (5,  "Penn Station-34 St",    40.7506, -73.9935, "Manhattan", "123ACE"),
    (6,  "Fulton St",             40.7091, -74.0064, "Manhattan", "2345ACJZ"),
    (7,  "59 St-Columbus Circle", 40.7681, -73.9819, "Manhattan", "1ABCD"),
    (8,  "42 St-Port Authority",  40.7573, -73.9901, "Manhattan", "ACENQRS1237W"),
    (9,  "Canal St",              40.7189, -74.0000, "Manhattan", "JNQRZ6W"),
    (10, "Wall St",               40.7069, -74.0090, "Manhattan", "23"),
    (11, "Chambers St",           40.7142, -74.0089, "Manhattan", "123ACE"),
    (12, "Astor Place",           40.7300, -73.9909, "Manhattan", "6"),
    (13, "Bleecker St",           40.7258, -73.9946, "Manhattan", "6BDF"),
    (14, "23 St (6)",             40.7396, -73.9865, "Manhattan", "6"),
    (15, "28 St (6)",             40.7432, -73.9845, "Manhattan", "6"),
    (16, "33 St (6)",             40.7468, -73.9827, "Manhattan", "6"),
    (17, "51 St",                 40.7572, -73.9718, "Manhattan", "6"),
    (18, "68 St-Hunter College",  40.7687, -73.9642, "Manhattan", "6"),
    (19, "77 St",                 40.7736, -73.9598, "Manhattan", "6"),
    (20, "86 St (456)",           40.7794, -73.9556, "Manhattan", "456"),
    (21, "96 St (6)",             40.7854, -73.9510, "Manhattan", "6"),
    (22, "103 St (6)",            40.7906, -73.9474, "Manhattan", "6"),
    (23, "W 4 St-Washington Sq",  40.7321, -74.0005, "Manhattan", "ABCDEFM"),
    (24, "Spring St (6)",         40.7223, -73.9967, "Manhattan", "6"),
    (25, "Brooklyn Bridge",       40.7134, -74.0040, "Manhattan", "456JZ"),
    (26, "Atlantic Av-Barclays",  40.6844, -73.9777, "Brooklyn",  "2345BDNQR"),
    (27, "Jay St-MetroTech",      40.6923, -73.9872, "Brooklyn",  "ACFR"),
    (28, "Bedford Av",            40.7174, -73.9567, "Brooklyn",  "L"),
    (29, "1 Av (L)",              40.7307, -73.9817, "Manhattan", "L"),
    (30, "3 Av (L)",              40.7327, -73.9862, "Manhattan", "L"),
    (31, "6 Av (L)",              40.7374, -73.9967, "Manhattan", "L"),
    (32, "8 Av (L)",              40.7399, -74.0025, "Manhattan", "L"),
    (33, "DeKalb Av",             40.6907, -73.9817, "Brooklyn",  "BDNQR"),
    (34, "Hoyt-Schermerhorn",     40.6884, -73.9851, "Brooklyn",  "ACG"),
    (35, "Bergen St",             40.6861, -73.9906, "Brooklyn",  "FG"),
    (36, "Court St",              40.6817, -73.9920, "Brooklyn",  "NR"),
    (37, "Borough Hall",          40.6926, -73.9900, "Brooklyn",  "2345R"),
    (38, "High St",               40.6994, -73.9908, "Brooklyn",  "AC"),
    (39, "Clark St",              40.6975, -73.9932, "Brooklyn",  "23"),
    (40, "Lorimer St",            40.7140, -73.9504, "Brooklyn",  "LG")
  )

  // ----- Real Citi Bike docks (subset of 50 key docks) -----
  val bikeDocks: Seq[(Int, String, Double, Double, Int)] = Seq(
    (1,  "E 16 St & Union Sq W",      40.7359, -73.9910, 39),
    (2,  "E 47 St & 2 Av",            40.7530, -73.9706, 31),
    (3,  "Allen St & Stanton St",     40.7224, -73.9898, 23),
    (4,  "W 21 St & 6 Av",            40.7418, -73.9940, 33),
    (5,  "Broadway & E 22 St",        40.7404, -73.9892, 27),
    (6,  "W 42 St & 8 Av",            40.7575, -73.9903, 39),
    (7,  "E 43 St & Vanderbilt Av",   40.7530, -73.9789, 27),
    (8,  "Lafayette St & E 8 St",     40.7306, -73.9912, 31),
    (9,  "1 Av & E 15 St",            40.7321, -73.9816, 35),
    (10, "6 Av & W 33 St",            40.7490, -73.9891, 41),
    (11, "Broadway & W 41 St",        40.7553, -73.9867, 39),
    (12, "W 37 St & 5 Av",            40.7516, -73.9844, 27),
    (13, "Broadway & W 36 St",        40.7513, -73.9879, 27),
    (14, "Grand St & Elizabeth St",   40.7186, -73.9963, 27),
    (15, "E 2 St & Av A",             40.7252, -73.9858, 19),
    (16, "Pershing Square North",     40.7519, -73.9778, 59),
    (17, "Broadway & W 49 St",        40.7606, -73.9840, 41),
    (18, "Park Av & E 33 St",         40.7468, -73.9821, 27),
    (19, "Rivington St & Chrystie St",40.7219, -73.9919, 23),
    (20, "E 13 St & Av A",            40.7296, -73.9812, 23),
    (21, "W 52 St & 11 Av",           40.7673, -73.9939, 41),
    (22, "E 32 St & Park Av",         40.7455, -73.9819, 31),
    (23, "W 56 St & 6 Av",            40.7638, -73.9773, 37),
    (24, "Elizabeth St & Hester St",   40.7173, -73.9964, 23),
    (25, "MacDougal St & Prince St",  40.7272, -74.0003, 23),
    (26, "Lexington Av & E 63 St",    40.7641, -73.9666, 31),
    (27, "Av A & E 3 St",             40.7260, -73.9841, 23),
    (28, "E 20 St & FDR Drive",       40.7338, -73.9758, 31),
    (29, "Duane St & Greenwich St",   40.7166, -74.0107, 23),
    (30, "Cooper Sq & Astor Pl",      40.7296, -73.9910, 23),
    (31, "Mercer St & Spring St",     40.7237, -73.9994, 31),
    (32, "W 20 St & 11 Av",           40.7473, -74.0079, 37),
    (33, "W 14 St & Hudson St",       40.7374, -74.0065, 27),
    (34, "E 39 St & 3 Av",            40.7497, -73.9760, 27),
    (35, "W 72 St & Central Park W",  40.7766, -73.9769, 39),
    (36, "W 84 St & Columbus Av",     40.7857, -73.9743, 31),
    (37, "E 55 St & Lexington Av",    40.7590, -73.9680, 31),
    (38, "Barclay St & Church St",    40.7129, -74.0102, 27),
    (39, "Sullivan St & Washington Sq",40.7306,-73.9999, 23),
    (40, "E 10 St & Av A",            40.7276, -73.9819, 23),
    (41, "W 82 St & Central Park W",  40.7840, -73.9718, 23),
    (42, "Kent Av & N 7 St",          40.7213, -73.9614, 27),
    (43, "Wythe Av & Metropolitan Av",40.7167, -73.9633, 23),
    (44, "Berry St & N 8 St",         40.7193, -73.9585, 19),
    (45, "N 6 St & Bedford Av",       40.7180, -73.9567, 23),
    (46, "Smith St & 3 St",           40.6789, -73.9958, 23),
    (47, "Court St & State St",       40.6902, -73.9919, 23),
    (48, "Clermont Av & Lafayette Av",40.6892, -73.9710, 19),
    (49, "Adelphi St & Myrtle Av",    40.6935, -73.9716, 23),
    (50, "Vanderbilt Av & Grand Av",  40.6882, -73.9688, 23)
  )

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("NeuroTraffic-SampleData")
      .getOrCreate()

    val target = if (args.nonEmpty) args(0) else "local"
    val basePath = target match {
      case "hdfs"  => "hdfs:///data"
      case "local" => "sample_data"
      case path    => path
    }

    println(s"Generating sample data to: $basePath")

    generateNodes(spark, basePath)
    generateTrips(spark, basePath)

    println("\nSample data generation complete!")
    println(s"Node files:  $basePath/nodes/")
    println(s"Trips file:  $basePath/combined/trips.parquet")

    spark.stop()
  }

  private def generateNodes(spark: SparkSession, basePath: String): Unit = {
    import spark.implicits._

    // Taxi zones
    val taxiDF = taxiZones.map { case (id, name, lat, lon, borough) =>
      (id, name, lat, lon, estimateZoneCapacity(name), borough)
    }.toDF("zone_id", "zone_name", "lat", "lon", "capacity", "borough")

    taxiDF.write.mode(SaveMode.Overwrite).parquet(s"$basePath/nodes/taxi_zones.parquet")
    println(s"  Taxi zones:      ${taxiDF.count()} nodes")

    // Subway stations
    val subwayDF = subwayStations.map { case (id, name, lat, lon, borough, lines) =>
      (id, name, lat, lon, estimateStationCapacity(lines), borough)
    }.toDF("station_id", "station_name", "lat", "lon", "capacity", "borough")

    subwayDF.write.mode(SaveMode.Overwrite).parquet(s"$basePath/nodes/subway_stations.parquet")
    println(s"  Subway stations: ${subwayDF.count()} nodes")

    // Bike docks
    val bikeDF = bikeDocks.map { case (id, name, lat, lon, cap) =>
      val borough = if (name.contains("Kent") || name.contains("Wythe") ||
        name.contains("Berry") || name.contains("Bedford") || name.contains("Smith") ||
        name.contains("Court St & State") || name.contains("Clermont") ||
        name.contains("Adelphi") || name.contains("Vanderbilt Av & Grand"))
        "Brooklyn" else "Manhattan"
      (id, name, lat, lon, cap, borough)
    }.toDF("dock_id", "dock_name", "lat", "lon", "capacity", "borough")

    bikeDF.write.mode(SaveMode.Overwrite).parquet(s"$basePath/nodes/bike_docks.parquet")
    println(s"  Bike docks:      ${bikeDF.count()} nodes")
  }

  private def generateTrips(spark: SparkSession, basePath: String): Unit = {
    import spark.implicits._

    val trips = scala.collection.mutable.ArrayBuffer[(Long, Long, String, String, Long, Double, Int, String)]()

    // --- Taxi-to-taxi trips (the bulk of edges) ---
    for (src <- taxiZones; dst <- taxiZones if src._1 != dst._1) {
      val dist = haversine(src._3, src._4, dst._3, dst._4)
      if (dist < 5.0) { // only generate edges for zones within 5 km
        for (hour <- Seq(8, 9, 12, 17, 18, 2); dayType <- Seq("weekday", "weekend")) {
          val baseDemand = if (isManhattan(src._5) && isManhattan(dst._5)) 200 else 50
          val rushMultiplier = if (hour >= 7 && hour <= 9 || hour >= 17 && hour <= 19) 3.0 else 1.0
          val weekendMultiplier = if (dayType == "weekend") 0.6 else 1.0
          val tripCount = (baseDemand * rushMultiplier * weekendMultiplier * (1.0 + rand.nextGaussian() * 0.3)).toLong.max(1)
          val travelTime = dist * 3.0 + rand.nextGaussian() * 2.0 // ~3 min per km + noise

          trips += ((src._1.toLong, dst._1.toLong, "taxi", "taxi",
            tripCount, travelTime.max(2.0), hour, dayType))
        }
      }
    }

    // --- Subway-to-taxi trips (people exiting subway, taking taxi) ---
    for (station <- subwayStations; zone <- taxiZones) {
      val dist = haversine(station._3, station._4, zone._3, zone._4)
      if (dist < 1.5) { // subway exits feed nearby taxi zones
        for (hour <- Seq(8, 9, 17, 18); dayType <- Seq("weekday", "weekend")) {
          val tripCount = (80 * (1.5 - dist) * (1.0 + rand.nextGaussian() * 0.2)).toLong.max(1)
          trips += ((station._1.toLong, zone._1.toLong, "subway", "taxi",
            tripCount, 5.0 + rand.nextGaussian(), hour, dayType))
        }
      }
    }

    // --- Taxi-to-subway trips (people taking taxi to subway) ---
    for (zone <- taxiZones; station <- subwayStations) {
      val dist = haversine(zone._3, zone._4, station._3, station._4)
      if (dist < 1.5) {
        for (hour <- Seq(7, 8, 17, 18); dayType <- Seq("weekday", "weekend")) {
          val tripCount = (60 * (1.5 - dist) * (1.0 + rand.nextGaussian() * 0.2)).toLong.max(1)
          trips += ((zone._1.toLong, station._1.toLong, "taxi", "subway",
            tripCount, 5.0 + rand.nextGaussian(), hour, dayType))
        }
      }
    }

    // --- Bike-to-bike trips ---
    for (src <- bikeDocks; dst <- bikeDocks if src._1 != dst._1) {
      val dist = haversine(src._4, src._5, dst._4, dst._5)
      if (dist < 3.0) {
        for (hour <- Seq(8, 9, 12, 17, 18); dayType <- Seq("weekday", "weekend")) {
          val tripCount = (30 * (3.0 - dist) * (1.0 + rand.nextGaussian() * 0.3)).toLong.max(1)
          val travelTime = dist * 5.0 + rand.nextGaussian() * 2.0 // ~5 min per km on bike
          trips += ((src._1.toLong, dst._1.toLong, "bike", "bike",
            tripCount, travelTime.max(3.0), hour, dayType))
        }
      }
    }

    // --- Subway-to-bike trips (exit subway, grab a bike) ---
    for (station <- subwayStations; dock <- bikeDocks) {
      val dist = haversine(station._3, station._4, dock._4, dock._5)
      if (dist < 0.8) { // bike docks very close to station exits
        for (hour <- Seq(8, 9, 17, 18); dayType <- Seq("weekday", "weekend")) {
          val tripCount = (40 * (0.8 - dist) * (1.0 + rand.nextGaussian() * 0.2)).toLong.max(1)
          trips += ((station._1.toLong, dock._1.toLong, "subway", "bike",
            tripCount, 3.0 + rand.nextGaussian(), hour, dayType))
        }
      }
    }

    // --- Bike-to-subway trips (ride bike to subway) ---
    for (dock <- bikeDocks; station <- subwayStations) {
      val dist = haversine(dock._4, dock._5, station._3, station._4)
      if (dist < 0.8) {
        for (hour <- Seq(7, 8, 17, 18); dayType <- Seq("weekday", "weekend")) {
          val tripCount = (35 * (0.8 - dist) * (1.0 + rand.nextGaussian() * 0.2)).toLong.max(1)
          trips += ((dock._1.toLong, station._1.toLong, "bike", "subway",
            tripCount, 3.0 + rand.nextGaussian(), hour, dayType))
        }
      }
    }

    val tripsDF = trips.toSeq.toDF(
      "src_node_id", "dst_node_id", "src_type", "dst_type",
      "trip_count", "avg_travel_time", "hour_of_day", "day_type"
    )

    tripsDF.write.mode(SaveMode.Overwrite).parquet(s"$basePath/combined/trips.parquet")

    println(s"\n  Total trips generated: ${tripsDF.count()}")
    println("  Edge type breakdown:")

    val crossModal = trips.groupBy(t => s"${t._3} -> ${t._4}")
    crossModal.toSeq.sortBy(-_._2.size).foreach { case (edgeType, edgeList) =>
      println(f"    $edgeType%-20s ${edgeList.size}%,d edges")
    }
  }

  // Haversine distance in kilometers
  private def haversine(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double = {
    val R = 6371.0
    val dLat = Math.toRadians(lat2 - lat1)
    val dLon = Math.toRadians(lon2 - lon1)
    val a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2)
    R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  }

  private def isManhattan(borough: String): Boolean = borough == "Manhattan"

  // Busier zones get higher capacity estimates
  private def estimateZoneCapacity(name: String): Int = {
    val highTraffic = Seq("Midtown", "Times Square", "Penn Station", "Financial District",
      "Garment District", "Flatiron", "Gramercy", "Murray Hill")
    if (highTraffic.exists(name.contains(_))) 5000 else 2000
  }

  // More subway lines = higher capacity
  private def estimateStationCapacity(lines: String): Int = {
    lines.length * 3000
  }
}
