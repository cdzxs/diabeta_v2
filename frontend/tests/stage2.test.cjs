// Local page-handler flow tests. Run: node frontend/tests/stage2.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../stage2.html'), 'utf8');
const render = html.slice(html.indexOf('            function renderStage2Result'),
    html.indexOf('            function populateFormFromCurrentAssessment'));
const save = html.slice(html.indexOf('            function saveCurrentDraft'), html.indexOf('            const TIER_META'));
const submit = html.slice(html.indexOf("            form.addEventListener('submit'"), html.indexOf('            /*', html.indexOf("form.addEventListener('submit'")));
const inputs = Object.fromEntries([...html.matchAll(/<input\b[\s\S]*?>/g)].map(([tag]) => {
    const attr = name => tag.match(new RegExp(name+'="([^"]*)"'))?.[1];
    return [attr('name'), {id:attr('id'), value:attr('value') || '', placeholder:attr('placeholder')}];
}));
assert.equal(inputs.fasting_glucose.id,'stage2Glucose');
assert.equal(inputs.fasting_glucose.value,'');
assert.doesNotMatch(inputs.fasting_glucose.placeholder,/^\d/);
const profile = {age:'45', sex:'2', bmi:'27', race_ethnicity:'3', family_history:'0',
    hypertension:'0', physical_activity:'1', smoking_status:'0', hba1c:'5.9', fasting_glucose:'96'};
const supplied = process.env.STAGE2_FLOW ? JSON.parse(process.env.STAGE2_FLOW) : null;
function response(glucose, high=false) {
    return {tier:high?'High':'Moderate',final_risk_level:high?'High':'Moderate',
        stage3_eligible:!high && glucose!==null, inputs_used:{hba1c:high?7:5.9,fasting_glucose:glucose},
        person:{Age:45,Sex:'Female',BMI:27,HbA1c:(high?'7':'5.9')+'%',
            'Fasting Glucose':glucose===null?'Not provided':glucose+' mg/dL'}};
}
function harness(initial, api, pending=null) {
    const fields=Object.fromEntries(Object.entries(initial).map(([k,v])=>[k,{value:v}]));
    const nodes={}, storage={}, calls=[];
    const node=id=>nodes[id]??={style:{},textContent:'',innerHTML:'',children:[],
        appendChild(child){this.children.push(child)},scrollIntoView(){}};
    let handler;
    const context={console,JSON,Number,String,Object,Error,Date,
        fieldRefs:fields, FORM_STATE_KEY:'draft', BACKEND_URL:'http://localhost:5000',
        TIER_META:{Moderate:{css:'moderate',title:'Moderate Risk'},High:{css:'high',title:'High Risk'}},
        traceAssessment(){},form:{addEventListener:(event,fn)=>handler=fn},
        document:{getElementById:node,createElement:()=>({})},
        localStorage:{getItem:k=>storage[k]??null,setItem:(k,v)=>storage[k]=v,removeItem:k=>delete storage[k]},
        FormData:class {constructor(){this.values=Object.fromEntries(Object.entries(fields).map(([k,v])=>[k,v.value]));}get(k){return this.values[k]??null;}},
        fetch:async(url,options)=>{calls.push(JSON.parse(options.body));if(pending) await pending;return {ok:true,json:async()=>api};},
        ...Object.fromEntries(['submitBtn','formError','resultCard','tierBadge','tierTitle','riskNumber','reasonList','factorsList'].map(k=>[k,node(k)]))};
    vm.createContext(context);
    vm.runInContext('let assessmentRevision=0;\n'+save+render+submit,context);
    return {fields,nodes,storage,calls,submit:()=>handler({preventDefault(){},target:{}}),
        edit:()=>vm.runInContext('saveCurrentDraft()',context),
        restore:record=>{context.cachedRecord=record;vm.runInContext('renderStage2Result(cachedRecord, cachedRecord.person)',context)}};
}
(async()=>{
    if (process.env.STAGE2_EXPLANATION) {
        const api=JSON.parse(process.env.STAGE2_EXPLANATION);
        const h=harness(profile,api);await h.submit();
        assert.equal(h.nodes.stage2RuleExplanation.textContent,api.final_recommendation.explanation);
        assert.equal(h.nodes.stage2NextStep.textContent,api.final_recommendation.next_step);
        assert.equal(h.nodes.stage2LabAssessment.children.length,2);
        for (let i=0;i<2;i++) assert.ok(h.nodes.stage2LabAssessment.children[i].textContent.includes(api.lab_assessment[i].finding));
        const record=JSON.parse(h.storage.diabeta_stage2_hybrid_result);
        assert.deepEqual(record.lab_assessment,api.lab_assessment);
        assert.deepEqual(record.final_recommendation,api.final_recommendation);
        assert.deepEqual(record.profile_screening,api.profile_screening);
        h.restore(record);
        assert.equal(h.nodes.stage2RuleExplanation.textContent,api.final_recommendation.explanation);
        assert.equal(h.nodes.stage2NextStep.textContent,api.final_recommendation.next_step);
        return;
    }
    const traces=[];
    for(const [name,glucose,high,api] of [
        ['HbA1c 5.9 without glucose','',false,supplied?.missing||response(null)],
        ['HbA1c 5.9 with entered glucose 96','96',false,supplied?.moderate||response(96)],
        ['genuine High','96',true,supplied?.high||response(96,true)]]) {
        const h=harness({...profile,fasting_glucose:glucose,hba1c:high?'7':'5.9'},api);
        h.storage.diabeta_last_result=JSON.stringify({tier:'Low',probability:12,person:{'Patient Name':'Previous synthetic patient'}});
        await h.submit();
        const summary=JSON.parse(h.storage.diabeta_last_result);
        assert.equal(summary.probability,undefined);
        assert.equal(summary.tier,undefined);
        assert.equal(summary.fromDirectStage2,true);
        const saved=JSON.parse(h.storage.diabeta_stage2_hybrid_result);
        assert.equal(h.calls[0].fasting_glucose,glucose||null);
        assert.deepEqual(saved.raw_inputs,h.calls[0]);
        assert.equal(saved.fasting_glucose,glucose?96:null);
        assert.equal(h.nodes.tierBadge.textContent,high?'High':'Moderate');
        assert.ok(h.nodes.factorsList.children.some(n=>n.innerHTML.includes('<dt>Fasting Glucose</dt><dd>'+(glucose?'96':'Not provided'))));
        const notice=h.nodes.stage3NotApplicableNotice;
        if(high) {assert.match(notice.textContent,/High-risk/);assert.equal(h.nodes.continueToStage3Btn.style.display,'none');}
        else if(!glucose) {assert.match(notice.textContent,/fasting glucose result is required/);assert.doesNotMatch(notice.textContent,/High/);}
        else {assert.equal(notice.style.display,'none');assert.equal(h.nodes.continueToStage3Btn.style.display,'');}
        traces.push(saved.raw_inputs);
        console.log('PASS '+name+' form → payload → response → display → storage');
        h.fields.fasting_glucose.value='105';h.edit();
        assert.equal(h.nodes.resultCard.style.display,'none');
        assert.equal(h.nodes.continueToStage3Btn.style.display,'none');
        assert.equal(h.storage.diabeta_stage2_hybrid_result,undefined);
        assert.equal(h.storage.diabeta_last_result,undefined);
    }
    let release;const pending=new Promise(r=>release=r);
    const h=harness(profile,response(96),pending);const request=h.submit();
    h.fields.fasting_glucose.value='120';h.edit();release();await request;
    assert.equal(h.storage.diabeta_stage2_hybrid_result,undefined);
    assert.match(h.nodes.formError.textContent,/Inputs changed/);
    console.log('PASS edited/in-flight inputs invalidate old results');
    const injected=response(96);
    injected.person['Patient Name']='<img src=x onerror=alert(1)>';
    const escaped=harness(profile,injected);await escaped.submit();
    assert.ok(escaped.nodes.factorsList.children.some(n=>n.innerHTML.includes('&lt;img')));
    assert.ok(escaped.nodes.factorsList.children.every(n=>!n.innerHTML.includes('<img')));
    console.log('PASS patient text is escaped in Stage 2 summary');
    if(supplied) console.log('FLOW_RESULT:'+JSON.stringify(traces));
})().catch(error=>{console.error(error);process.exitCode=1});
