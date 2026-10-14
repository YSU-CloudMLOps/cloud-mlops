import React, { useState } from 'react';
import { Box, Layers, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';

export const EquipmentInfo: React.FC = () => {
  const [expanded, setExpanded] = useState<boolean>(false);

  return (
    <div className="card equipment-info-card">
      <div className="card-header">
        <div>
          <span className="eyebrow-mono">// SYSTEM CONTEXT &amp; PRODUCT SCOPE</span>
          <div className="card-title-group">
            <Box size={18} className="card-icon" />
            <h2 className="card-title">예측 대상 설비 및 생산 제품군 안내</h2>
          </div>
        </div>
        <button
          type="button"
          className="btn-outline-pill btn-toggle-detail"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
        >
          <span>{expanded ? '상세 설명 접기' : '상세 가공 공정 및 5대 고장 보기'}</span>
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>
      </div>

      <p className="card-description">
        본 AI 시스템은 <strong>스마트 CNC 밀링 머신(CNC Milling Machine)</strong>에서 고속 절삭 회전(Spindle)을 통해 가공되는 <strong>3대 산업용 부품(L/M/H)</strong>의 고장 위험을 실시간 진단합니다.
      </p>

      {/* Product Variants 3-Column Grid */}
      <div className="product-types-grid">
        <div className="product-type-box">
          <div className="product-type-badge-row">
            <span className="type-pill-lg type-l">L TYPE</span>
            <span className="product-share-badge">전체 생산의 60%</span>
          </div>
          <h3 className="product-box-title">일반·양산형 표준 부품 (Low Variant)</h3>
          <p className="product-box-desc">
            범용 기계 부품 및 일반 소모품 등 대량 양산되는 표준 제품입니다. 단가와 가공 속도 위주로 설계되어 허용 스트레인 한계(11,000 min·N·m)가 낮고, 열과 마모에 가장 민감하게 반응합니다.
          </p>
        </div>

        <div className="product-type-box">
          <div className="product-type-badge-row">
            <span className="type-pill-lg type-m">M TYPE</span>
            <span className="product-share-badge">전체 생산의 30%</span>
          </div>
          <h3 className="product-box-title">산업·자동차 중급형 부품 (Medium Variant)</h3>
          <p className="product-box-desc">
            자동차 동력 전달 계통, 유압 밸브, 중공업 기계에 장착되는 표준 부품입니다. 중간 수준의 열 분산성과 과부하 임계치(12,000 min·N·m)를 가지며 균형 잡힌 가공 안전성을 요구합니다.
          </p>
        </div>

        <div className="product-type-box">
          <div className="product-type-badge-row">
            <span className="type-pill-lg type-h">H TYPE</span>
            <span className="product-share-badge">전체 생산의 10%</span>
          </div>
          <h3 className="product-box-title">항공·의료기기 고정밀 부품 (High Variant)</h3>
          <p className="product-box-desc">
            항공우주 터빈 부품, 반도체 장비 부품, 정밀 수술 도구 등 마이크로미터 단위 공차가 요구되는 고부가가치 제품입니다. 엄격한 품질 규격과 최고 수준의 과부하 허용치(13,000 min·N·m)가 적용됩니다.
          </p>
        </div>
      </div>

      {/* Expanded Details: Equipment Mechanics & 5 Failure Modes */}
      {expanded && (
        <div className="equipment-details-drawer">
          <div className="details-grid">
            <div className="detail-col">
              <div className="detail-col-header">
                <Layers size={14} />
                <span className="detail-col-title">// CNC 밀링 머신(Milling Machine)이란?</span>
              </div>
              <p className="detail-col-body">
                고속으로 회전하는 스핀들(Spindle)에 절삭 공구(End Mill/Cutter)를 물리고, 3축(X/Y/Z)으로 공작물을 이동시키며 금속 모재를 깎아내는 정밀 가공 설비입니다.
                가공 중 <strong>회전 속도(RPM)</strong>와 <strong>토크(Torque)</strong>, 절삭 마찰로 인한 <strong>공정 온도</strong>, 그리고 <strong>공구 누적 마모 시간</strong>이 복합적으로 작용하여 설비 고장으로 이어집니다.
              </p>
            </div>

            <div className="detail-col">
              <div className="detail-col-header">
                <AlertCircle size={14} />
                <span className="detail-col-title">// 제품 가공 중 판별하는 5대 고장 모드</span>
              </div>
              <ul className="failure-modes-list">
                <li>
                  <code>HDF (방열 고장)</code> 공정온도와 대기온도 차(ΔT &lt; 8.6K)가 좁고 저속 회전 시 냉각 불량으로 과열 발생
                </li>
                <li>
                  <code>PWF (전력 이상)</code> 주축 소비 전력이 적정 범위(3.5kW ~ 9.0kW)를 이탈하여 모터 과부하 또는 공회전 발생
                </li>
                <li>
                  <code>OSF (과부하 파손)</code> 누적 마모된 공구에 과도한 토크가 인가되어 제품별 허용 스트레인 한계를 초과할 때 발생
                </li>
                <li>
                  <code>TWF (공구 마모)</code> 공구 마모 시간이 200~240분의 피로 임계점에 도달하여 절삭 날이 파손
                </li>
                <li>
                  <code>RNF (무작위 외생)</code> 센서 계측과 무관한 외부 돌발 요인으로 인한 0.1% 미만의 우발적 고장
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
