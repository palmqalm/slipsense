import React, { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import * as tf from "@tensorflow/tfjs";
import { MapContainer, TileLayer, Marker, Popup, Rectangle } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "bootstrap/dist/css/bootstrap.min.css";

function parseLocalDate(dateStr, timeStr) {
  try {
    if (!dateStr) return null;
    const sdate = dateStr.toString().trim();
    const stime = (timeStr || "00:00:00").toString().trim();
    const t = stime.replace(/\./g, ":");
    const d = sdate.includes("/") ? sdate.split("/") : sdate.split("-");
    let day, month, year;
    if (d.length === 3) {
      if (+d[0] > 31) { year = +d[0]; month = +d[1]; day = +d[2]; }
      else { day = +d[0]; month = +d[1]; year = +d[2]; }
    } else return new Date(sdate + " " + t);
    if (year > 2400) year -= 543;
    if (year < 100) year += 2000;
    const [hh, mm, ss] = t.split(":").map((x) => parseInt(x || "0", 10));
    return new Date(year, month - 1, day, hh, mm, ss);
  } catch (e) { return null; }
}

function haversineKm(aLat, aLon, bLat, bLon) {
  const R = 6371;
  const toRad = (deg) => (deg * Math.PI) / 180;
  const dLat = toRad(bLat - aLat);
  const dLon = toRad(bLon - aLon);
  const lat1 = toRad(aLat);
  const lat2 = toRad(bLat);
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function groupByGrid(rows, cellKm = 50) {
  const degLat = cellKm / 111;
  const grid = new Map();
  rows.forEach((r) => {
    const gy = Math.floor(r.lat / degLat);
    const gx = Math.floor(r.lon / (degLat * Math.cos((r.lat * Math.PI) / 180) || 1));
    const key = `${gx},${gy}`;
    if (!grid.has(key)) grid.set(key, { gx, gy, degLat, rows: [] });
    grid.get(key).rows.push(r);
  });
  return grid;
}

export default function App() {
  const [localData, setLocalData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [nextPrediction, setNextPrediction] = useState(null);
  const [cellKm, setCellKm] = useState(50);
  const [lookbackDays, setLookbackDays] = useState(30);
  const [horizonDays, setHorizonDays] = useState(7);
  const [magThreshold, setMagThreshold] = useState(4.0);
  const [trainingStatus, setTrainingStatus] = useState("");

  const handleFile = (evt) => {
    const file = evt.target.files && evt.target.files[0];
    if (!file) return;
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        const rows = res.data
          .map((r) => {
            const dt = parseLocalDate(r["DATE"], r["TIME"]);
            const lat = parseFloat((r["LAT"] || r["Latitude"] || "").toString().replace(",", "."));
            const lon = parseFloat((r["LONG"] || r["LON"] || r["Longitude"] || "").toString().replace(",", "."));
            const mag = parseFloat((r["M/I"] || r["MAG"] || r["M"] || "").toString().replace(",", "."));
            if (!dt || Number.isNaN(lat) || Number.isNaN(lon) || Number.isNaN(mag)) return null;
            return { date: dt, lat, lon, mag, locate: r["LOCATE"] || r["PLACE"] || "-" };
          })
          .filter(Boolean)
          .sort((a, b) => a.date - b.date);
        setLocalData(rows);
      },
    });
  };

  const buildDataset = (rows) => {
    const grid = groupByGrid(rows, cellKm);
    const X = [], y = [], meta = [];
    grid.forEach((cell) => {
      const cellRows = cell.rows.sort((a, b) => a.date - b.date);
      for (let i = 0; i < cellRows.length - 1; i++) {
        const cur = cellRows[i];
        const t = cur.date.getTime();
        const lookStart = t - lookbackDays * 24 * 3600 * 1000;
        const past = cellRows.filter((r) => r.date.getTime() >= lookStart && r.date.getTime() < t);
        const cnt = past.length;
        const avgMag = cnt ? past.reduce((s, e) => s + e.mag, 0) / cnt : 0;
        const lastTime = past.length ? (t - past[past.length - 1].date.getTime()) / 3600000 : 999999;
        const horizonEnd = t + horizonDays * 24 * 3600 * 1000;
        const futureEvents = cellRows.filter((r) => r.date.getTime() > t && r.date.getTime() <= horizonEnd);
        const label = futureEvents.some((fe) => fe.mag >= magThreshold) ? 1 : 0;
        X.push([cur.lat, cur.lon, cnt, avgMag, lastTime]);
        y.push(label);
        meta.push({ cell: `${cell.gx},${cell.gy}`, time: t, source: cur.locate });
      }
    });
    return { X, y, meta };
  };

  const trainAndPredict = async () => {
    try {
      setTrainingStatus("กำลังเตรียมข้อมูล...");
      const rows = localData;
      if (!rows || rows.length < 30) {
        alert("ข้อมูลไม่เพียงพอ (แนะนำอย่างน้อย 30 แถว)");
        setTrainingStatus("");
        return;
      }

      const { X, y, meta } = buildDataset(rows);
      if (X.length < 30) {
        alert("ตัวอย่างสำหรับเทรนนิ่งน้อยเกินไปหลังการประมวลผล");
        setTrainingStatus("");
        return;
      }

      setTrainingStatus("กำลังสร้างและเทรนโมเดล (TensorFlow.js)...");

      const xTensor = tf.tensor2d(X);
      const yTensor = tf.tensor2d(y, [y.length, 1]);
      const { mean, variance } = tf.moments(xTensor, 0);
      const std = tf.sqrt(variance).add(1e-6);
      const xNorm = xTensor.sub(mean).div(std);

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 32, activation: "relu", inputShape: [X[0].length] }));
      model.add(tf.layers.dropout({ rate: 0.2 }));
      model.add(tf.layers.dense({ units: 16, activation: "relu" }));
      model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
      model.compile({ optimizer: tf.train.adam(0.01), loss: "binaryCrossentropy", metrics: ["accuracy"] });

      await model.fit(xNorm, yTensor, { epochs: 40, batchSize: 32, verbose: 0 });

      setTrainingStatus("ประมวลผลการทำนายสำหรับแต่ละเหตุการณ์...");

      const preds = [];
      for (let i = 0; i < rows.length; i++) {
        const r = rows[i];
        const t = r.date.getTime();
        const lookStart = t - lookbackDays * 24 * 3600 * 1000;
        const past = rows.filter((e) => e.date.getTime() >= lookStart && e.date.getTime() < t && haversineKm(e.lat, e.lon, r.lat, r.lon) <= cellKm);
        const cnt = past.length;
        const avgMag = cnt ? past.reduce((s, e) => s + e.mag, 0) / cnt : 0;
        const lastTime = past.length ? (t - past[past.length - 1].date.getTime()) / 3600000 : 999999;
        const feat = tf.tensor2d([[r.lat, r.lon, cnt, avgMag, lastTime]]);
        const featNorm = feat.sub(mean).div(std);
        const prob = (await model.predict(featNorm).data())[0];
        preds.push({ ...r, probability: prob });
        feat.dispose(); featNorm.dispose();
      }

      setPredictions(preds.reverse());
      const sorted = [...preds].sort((a, b) => b.probability - a.probability);
      setNextPrediction(sorted[0] || null);

      setTrainingStatus("เสร็จสิ้น: โมเดลฝึกเสร็จและสร้างการคาดการณ์แล้ว");
      xTensor.dispose(); yTensor.dispose(); xNorm.dispose(); mean.dispose(); variance.dispose(); std.dispose();
    } catch (err) {
      console.error(err);
      setTrainingStatus("เกิดข้อผิดพลาดในการฝึกโมเดล (ดู console)");
    }
  };

  const mapCenter = useMemo(() => {
    if (localData.length) return [localData[localData.length - 1].lat, localData[localData.length - 1].lon];
    return [15.8700, 100.9925];
  }, [localData]);

  return (
    <div className="container my-4">
      <div className="d-flex flex-column flex-md-row justify-content-between align-items-start mb-3">
        <div>
          <h2>Slipsense — ระบบอัจฉริยะสำหรับตรวจจับและพยากรณ์การเคลื่อนตัวตามรอยเลื่อน</h2>
          <small className="text-muted">เครื่องมือเพื่อการวิเคราะห์เชิงสถิติ (ทดลอง) — ผลลัพธ์ไม่ใช่การเตือนภัยอย่างเป็นทางการ</small>
        </div>
      </div>

      <div className="card mb-3 p-3">
        <div className="row g-2 align-items-center">
          <div className="col-auto">
            <div className="mt-2 mt-md-0 text-muted">อัพโหลดไฟล์ CSV. ข้อมูลการเกิดแผ่นไหวย้อนหลัง</div>
            <input type="file" accept=".csv" className="form-control" onChange={handleFile} />
          </div>
          <div className="col-auto">
            <div className="mt-2 mt-md-0 text-muted">ขนาดกริด (km)</div>
            <input type="number" className="form-control" value={cellKm} onChange={(e) => setCellKm(+e.target.value)} placeholder="ขนาดกริด (km)" />
          </div>
          <div className="col-auto">
            <div className="mt-2 mt-md-0 text-muted">ย้อนหลัง (วัน)</div>
            <input type="number" className="form-control" value={lookbackDays} onChange={(e) => setLookbackDays(+e.target.value)} placeholder="ย้อนหลัง (วัน)" />
          </div>
          <div className="col-auto">
            <div className="mt-2 mt-md-0 text-muted">คาดการณ์ (วัน)</div>
            <input type="number" className="form-control" value={horizonDays} onChange={(e) => setHorizonDays(+e.target.value)} placeholder="คาดการณ์ (วัน)" />
          </div>
          <div className="col-auto">
            <div className="mt-2 mt-md-0 text-muted">เกณฑ์ M</div>
            <input type="number" step="0.1" className="form-control" value={magThreshold} onChange={(e) => setMagThreshold(+e.target.value)} placeholder="เกณฑ์ M" />
          </div>
          <div className="col-auto">
            <button className="btn btn-warning" onClick={trainAndPredict}>ฝึกสอนและคาดการณ์</button>
          </div>
        </div>
        <div className="mt-2"><small className="text-muted">สถานะ: {trainingStatus || "ยังไม่ได้ฝึกสอน"}</small></div>
      </div>

      {nextPrediction && (
        <div className="alert alert-warning">
          <h5>การคาดการณ์ครั้งถัดไป (ตำแหน่งที่มีความน่าจะเป็นสูงสุด)</h5>
          <p><strong>พื้นที่:</strong> {nextPrediction.locate || "-"}</p>
          <p><strong>พิกัด:</strong> {nextPrediction.lat.toFixed(4)}, {nextPrediction.lon.toFixed(4)}</p>
          <p><strong>Magnitude ล่าสุด:</strong> {nextPrediction.mag}</p>
          <p><strong>ความน่าจะเป็น ≥ M {magThreshold}:</strong> {(nextPrediction.probability * 100).toFixed(2)}%</p>
        </div>
      )}

      <div className="row g-3">
        <div className="col-lg-6">
          <div className="card p-2 h-100">
            <h5 className="card-title">แผนที่เหตุการณ์และความเสี่ยง</h5>
            <div className="map-wrap" style={{ height: "500px" }}>
              <MapContainer center={mapCenter} zoom={6} style={{ height: "100%", width: "100%" }}>
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                {localData.slice(-1000).map((d, i) => (
                  <Marker key={`m-${i}`} position={[d.lat, d.lon]}>
                    <Popup>
                      <div>
                        <div><strong>สถานที่:</strong> {d.locate}</div>
                        <div><strong>วันที่:</strong> {d.date.toLocaleString()}</div>
                        <div><strong>ขนาด:</strong> M {d.mag}</div>
                      </div>
                    </Popup>
                  </Marker>
                ))}
                {predictions.slice(0, 500).map((p, i) => (
                  <Rectangle
                    key={`r-${i}`}
                    bounds={[[p.lat - 0.08, p.lon - 0.08], [p.lat + 0.08, p.lon + 0.08]]}
                    pathOptions={{ fillOpacity: 0.12, color: `rgba(255,0,0,${Math.min(0.9, p.probability)})` }}
                  />
                ))}
              </MapContainer>
            </div>
          </div>
        </div>

        <div className="col-lg-6">
          <div className="card p-2 h-100">
            <h5 className="card-title">ตารางข้อมูลและการคาดการณ์ (ล่าสุดก่อน)</h5>
            <div className="table-responsive" style={{ maxHeight: "500px", overflowY: "auto" }}>
              <table className="table table-sm table-hover table-bordered">
                <thead className="table-dark">
                  <tr>
                    <th>วันที่</th>
                    <th>สถานที่</th>
                    <th>Lat</th>
                    <th>Lon</th>
                    <th>Magnitude</th>
                    <th>ความน่าจะเป็น (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((r, i) => (
                    <tr key={i}>
                      <td>{r.date.toLocaleString()}</td>
                      <td>{r.locate}</td>
                      <td>{r.lat.toFixed(4)}</td>
                      <td>{r.lon.toFixed(4)}</td>
                      <td>{r.mag}</td>
                      <td>{(r.probability * 100).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="text-muted mt-3" style={{ fontSize: 12 }}>
        หมายเหตุ: ผลการคาดการณ์เป็นการวิเคราะห์เชิงสถิติ ใช้เพื่อการศึกษาและประเมินความเสี่ยงเท่านั้น ไม่ใช่การเตือนภัยอย่างเป็นทางการ
      </div>
    </div>
  );
}
